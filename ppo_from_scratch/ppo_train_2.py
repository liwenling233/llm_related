from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelscope import snapshot_download

# ==============================================================================
# 模块 1: 数据准备
# 作用：将人类的自然语言问题，包装成模型能听懂的格式（Input IDs）
# ==============================================================================
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []

        # 预处理：把文本转换成 tokenizer 后的 input_ids 之前的文本格式
        for prompt in prompts:
            if apply_chat_template:
                # 如果是对话模型（如 Qwen-Chat），需要套上特定的对话模板
                # 例如：<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                # 否则简单加上开始符
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.final_prompts[index]

# ==============================================================================
# 模块 2: 评论家模型 (Critic)
# 作用：它不生成文本，而是给每一个 Token 的状态打分（Value Estimation）
# ==============================================================================
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 使用 Actor 的底座（Transformer部分）来理解语义
        self.base_model = base_model
        # 冻结底座参数（通常做法），或者设置很小的学习率，这里直接设为 eval 模式
        self.base_model.eval()
        # 核心差异：加一个线性层，把 1024/768 维的隐藏状态压缩成 1 个数字（分数）
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, num_actions):
        """
        Input: 完整的句子 (Prompt + Answer)
        Output: Answer 部分每一个 token 对应的预估分数
        """
        # 1. 过一遍 Transformer，拿到语义向量
        # shape: (batch_size, seq_len, hidden_size)
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        
        # 2. 过线性层，变成数值
        # shape: (batch_size, seq_len, 1)
        value_model_output = self.value_head(hidden_state)
        
        # 3. 裁剪。squeeze去掉最后一维，然后切片。
        # [:, :-1] 的意思是：最后一个 token 后面没有内容了，预测它的价值没意义（或者说它的价值由结束状态决定），通常去掉。
        # [:, -num_actions:] 的意思是：我们只关心“学生生成的回答部分”的价值，不关心“老师出的题目(Prompt)”的价值。
        values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
        return values

# ==============================================================================
# 模块 3: 损失函数 (PPO 核心数学公式)
# ==============================================================================
def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    """
    计算 PPO 的 Actor Loss。
    目标：鼓励那些 Advantage > 0 的动作，抑制 Advantage < 0 的动作，但别改太猛。
    """
    # 1. 计算新旧策略的比率 (Ratio)
    # log(a/b) = log(a) - log(b)  ==>  a/b = exp(log(a) - log(b))
    # ratio > 1 代表新策略更倾向于做这个动作
    ratio = (log_probs - old_log_probs).exp()
    
    # 2. 计算第一种 Loss：直接用优势加权
    surr1 = ratio * advantages
    
    # 3. 计算第二种 Loss：截断 (Clipping)
    # 如果 ratio 跑到了 (0.8, 1.2) 之外，就强制锁死在边界上。
    # 这就是 PPO 防止“步子迈太大扯着蛋”的机制。
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages    
    
    # 4. 取最小值 (Min)
    # 为什么要取最小？这是一种悲观保守策略（Lower Bound Optimization）。
    # 加上负号是因为 Pytorch 是做梯度下降（Minimize），而我们要最大化奖励。
    loss = -torch.min(surr1, surr2)
    
    # 5. Masking
    # Padding 的部分（无意义的补位符）不应该产生 Loss，所以要用 action_mask 过滤掉
    if action_mask is None:
        return loss.mean(-1).mean()
    # 分子：有效位置的 Loss 求和； 分母：有效位置的个数
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    """
    计算 Critic Loss (均方误差 MSE)。
    目标：让 Critic 预测的分数（values）尽可能接近真实的收益（returns）。
    """
    if clip_eps is not None:
        # PPO 的一个变种技巧：也限制 Critic 的更新幅度，防止 Critic 突然“发疯”
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        # 标准做法：预测值和真实值的差的平方
        loss = (values - returns) ** 2

    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

# ==============================================================================
# 模块 4: 经验池与数据结构
# ==============================================================================
class ExperienceBuffer:
    """一个简单的队列，存放采样回来的数据，供训练时反复抽取"""
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        # 将 dataclass 对象转为字典列表，方便处理
        batch = [{} for _ in range(len(experiences))]
        keys = ("seqs", "action_log_probs", "values", "returns", "advantages", "attention_mask", "action_mask", "num_actions")
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        
        self.buffer.extend(batch)
        # 保持 Buffer 不超过限制大小，模拟 FIFO
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
        
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]
    

# 定义两个数据结构，用于在函数间传递数据，避免参数列表过长
@dataclass
class Samples:
    """存放 rollout 阶段产生的原始数据"""
    seqs: torch.Tensor             # Prompt + Generated Text
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor] # 标记哪部分是生成的有效内容
    num_actions: Union[int, torch.Tensor]   # 生成了多少个 Token
    packed_seq_lens:Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor

@dataclass
class Experience:
    """存放经过计算（GAE, Reward）后的完整训练数据"""
    seqs: torch.Tensor
    action_log_probs: torch.Tensor # 旧策略的概率
    values: torch.Tensor           # 旧的 Critic 预测
    returns: Optional[torch.Tensor] # 真实回报 (Target Value)
    advantages: Optional[torch.Tensor] # 优势 (用于 Policy Update)
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None

# ==============================================================================
# 模块 5: 辅助计算函数 (GAE & KL)
# ==============================================================================
def compute_approx_kl(log_probs, ref_log_probs, action_mask=None):
    """
    计算 KL 散度的近似值 (log_ratio)。
    KL(P||Q) ≈ log(P) - log(Q)
    """
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio

def get_advantages_and_returns(values, rewards, action_mask, gamma, lambd):
    """
    ★ GAE (Generalized Advantage Estimation) 核心实现 ★
    这是 RL 中最难理解的部分。简单来说，它在平衡“短期收益”和“长期预测”。
    """
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)

    # 屏蔽掉 Padding 部分
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    # 从后往前遍历 (倒序)
    # 因为当前的优势取决于未来的收益，所以倒着算最方便
    for t in reversed(range(response_length)):
        # 如果是最后一步，next_value 就是 0；否则是 t+1 时刻的 value
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        
        # Delta = 真实奖励 + 折扣后的未来价值 - 当前预估价值
        # 这是“现实”与“预期”的差距
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        
        # GAE 公式：当前优势 = Delta + 衰减系数 * 下一步的优势
        # 这里的 lambd 决定了我们多大程度上信任 Critic 的长期预测
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    
    # 把倒序的结果翻转回来
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # Return = Advantage + Value (数学恒等式)
    returns = advantages + values
    return advantages.detach(), returns

# ==============================================================================
# 模块 6: 采样与推理 (Rollout)
# ==============================================================================
def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    """
    步骤 1: 让 Actor 模型做题 (Inference)
    """
    samples_list = []
    model.eval() # 推理模式
    
    # 将 Prompt 复制 n 份 (一个问题生成多个回答)
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], [])
    
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        batch_prompts = all_prompts[i:i+micro_rollout_batch_size]
        
        # 1. 编码输入
        # 这里的 padding='max_length' 很关键，保证 batch 内维度对齐
        inputs = actor_tokenizer(batch_prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # 2. 模型生成
        # seqs 包含了 [Prompt, Answer]
        seqs = model.generate(
            **inputs.to(device), 
            max_new_tokens=max_new_tokens, 
            eos_token_id=eos_token_id, 
            pad_token_id=pad_token_id
        )
        
        # 3. 长度对齐处理
        # 这一步是为了防止生成长度参差不齐，强行补 Pad 或截断，保证后续 Tensor 运算不报错
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)), fill_value=pad_token_id, device=seqs.device)], dim=1)

        # 4. 生成 Mask
        # attention_mask: 整个序列哪部分不是 pad
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        
        # 提取回答部分 (Answer)
        ans = seqs[:, input_ids.size(1):]
        # action_mask: 回答部分哪部分是有效的 (不是 eos 也不是 pad)
        # 只有这部分才计算 Reward 和 Loss
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list

def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    """
    计算最终奖励：Reward = 真实分数 - KL惩罚
    """
    # 1. 计算 KL 惩罚项 (通常 KL 是正的，这里取负号作为惩罚)
    kl_divergence_estimate = -kl_ctl * kl
    rewards = kl_divergence_estimate

    # 2. 将 Reward Model 的分数加进去
    # 注意：Reward Model 通常只给整句打一个分。
    # 我们通常把这个分加在生成的最后一个有效 token 上。
    ends = action_mask.sum(1) + 1 # 找到最后一个有效 token 的位置
    
    # 裁剪 Reward 防止分数过大导致梯度爆炸
    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)
    
    batch_size = r.size(0)
    for j in range(batch_size):
        # 只有在句子结束的地方加上 task reward，其他中间步骤只承担 KL 惩罚
        # (这是一种 sparse reward 的设定)
        if ends[j] < rewards.shape[1]: 
             rewards[j, ends[j]-1] += reward_clip[j, 0]
        else:
             rewards[j, -1] += reward_clip[j, 0]

    return rewards

def generate_experiences(samples_list):
    """
    步骤 2: 生成经验 (计算所有的概率、分数、优势)
    """
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    
    for samples in samples_list:
        seqs = samples.seqs
        # ... (解包 mask 等信息)
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        with torch.no_grad(): # 全程不需要梯度
            # 1. Actor 计算当前概率
            output = actor_model(seqs, attention_mask=attention_mask)
            # logits shape: (batch, seq_len, vocab_size)
            # 这里的 [:, :-1, :] 是 shift操作：第 i 个 token 的输出是为了预测 第 i+1 个 token
            log_probs = F.log_softmax(output.logits[:, :-1, :], dim=-1)
            # gather: 提取实际生成的那个 token 的概率
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

            # 2. Reference Model 计算参考概率 (用于 KL)
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

            # 3. Critic 计算状态价值
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            
            # 4. Reward Model 打分
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            r = reward_model(**reward_model_inputs.to(device)).logits
            
            # 5. 计算 KL 和 最终 Reward
            kl = compute_approx_kl(action_log_probs, ref_action_log_probs, action_mask=action_mask).to(device)
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=5.0) # clip值稍微调大一点适应实际
            
            # 6. GAE 计算优势
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.99, lambd=0.95)

        # 打包成 Experience 对象
        experiences.append(Experience(
            seqs, action_log_probs.detach(), value.detach(), returns.detach(),
            advantages.detach(), attention_mask, action_mask, r.detach(),
            samples.response_length, samples.total_length, num_actions, kl.detach()
        ))

    return experiences

# ==============================================================================
# 模块 7: 训练循环 (Training Loop)
# ==============================================================================
@dataclass
class BufferItem:
    """Collate_fn 返回的数据结构，把 list stack 成 tensor"""
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

def collate_fn(batch):
    # 将 batch 内的数据堆叠 (Stack/Cat) 起来
    # 代码省略重复部分，核心就是 torch.cat(dim=0)
    # ... (与源代码逻辑一致)
    seqs = torch.cat([x['seqs'] for x in batch], dim=0)
    # ... 其他字段同理
    action_log_probs = torch.cat([x['action_log_probs'] for x in batch], dim=0)
    values = torch.cat([x['values'] for x in batch], dim=0)
    returns = torch.cat([x['returns'] for x in batch], dim=0)
    advantages = torch.cat([x['advantages'] for x in batch], dim=0)
    attention_mask = torch.cat([x['attention_mask'] for x in batch], dim=0)
    action_mask = torch.cat([x['action_mask'] for x in batch], dim=0)

    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))
    
def train_step(experience, steps):
    """
    步骤 3: 真正更新参数
    """
    # 1. 训练 Actor
    actor_model.train()
    optimizer_actor.zero_grad()
    
    # 这一步非常重要：重新前向传播！
    # 因为我们需要新的梯度图 (Computation Graph)
    logits = actor_model(experience.seqs, attention_mask=experience.attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=experience.seqs[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -experience.num_actions:]
    
    # 计算 Actor Loss (PPO Loss)
    # 这里的 old_action_log_probs 来自 experience (固定值)，而 action_log_probs 来自上面的计算 (带梯度)
    policy_loss = compute_policy_loss(action_log_probs, experience.action_log_probs, experience.advantages, action_mask=experience.action_mask)
    policy_loss.backward()
    optimizer_actor.step()
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    # 2. 训练 Critic
    critic_model.train()
    optimizer_critic.zero_grad()
    # 重新计算 value
    values = critic_model.forward(experience.seqs, experience.attention_mask, experience.num_actions)
    # 计算 Critic Loss (MSE)
    value_loss = compute_value_loss(values, experience.values, experience.returns, experience.action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")

def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    
    # 外层循环：Episodes (多轮采样+训练)
    for episode in range(episodes):
        # 1. 遍历 Prompt 数据集
        for rand_prompts in prompts_dataloader:
            # 2. 采样：Actor 生成回答
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            
            # 3. 评估：计算奖励、优势，打包成经验
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            
            # 4. 训练：从经验池里把数据拿出来更新模型
            # 为什么要有 micro_train_batch_size？因为显存有限，要把大切片切碎了训练
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            
            torch.cuda.empty_cache() # 清理显存
            
            # PPO 允许利用同一批数据训练多次 (max_epochs)
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            
            buffer.clear() # PPO 是 On-Policy 算法，用完的数据必须扔掉！
            torch.cuda.empty_cache()    

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 配置区
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    episodes = 3              # 总共跑几轮
    max_epochs = 5            # 每一批数据反复学几次 (PPO特性)
    rollout_batch_size = 8    # 取出多少个 Prompt
    micro_rollout_batch_size = 2 # 显存不够时，分批生成
    n_samples_per_prompt = 2  # 每个问题生成几个回答
    max_new_tokens = 50       # 生成长度
    max_length = 256          # 总长度限制
    micro_train_batch_size = 2 # 训练时的 Batch Size

    writer = SummaryWriter('./runs') # Tensorboard 日志

    # 1. 加载 Actor 模型 (学生)
    actor_model = AutoModelForCausalLM.from_pretrained("D:\Pretrained_models\Qwen\Qwen2___5-0___5B-Instruct").to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained("D:\Pretrained_models\Qwen\Qwen2___5-0___5B-Instruct")
    actor_tokenizer.padding_side = 'left' # 生成任务通常左填充，为了让生成的词对齐
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id

    # 2. 加载 Reference Model (教科书)
    ref_model = AutoModelForCausalLM.from_pretrained("D:\Pretrained_models\Qwen\Qwen2___5-0___5B-Instruct").to(device)

    # 3. 加载 Reward Model (判卷老师)
    # 注意：这里用的是 Deberta，这是一个 BERT 类模型，用于分类/回归
    reward_model = AutoModelForSequenceClassification.from_pretrained("D:\Pretrained_models\deepset\deberta-v3-large-squad2").to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained("D:\Pretrained_models\deepset\deberta-v3-large-squad2")

    # 4. 初始化 Critic (助教)
    critic_model = Critic(actor_model.base_model).to(device)
    
    # 5. 优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=1e-5) # Actor 学习率通常很低
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=1e-5)
    
    # 6. 数据集
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
        ] # 示例 Prompt
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
    
    # 7. 开始训练
    train()