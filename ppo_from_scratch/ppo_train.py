from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelscope import snapshot_download

# 文件说明：
# 这是一个基于 PPO（近端策略优化）思想实现的训练脚本。
# 主要组件：
# - PromptDataset: 将文本提示转换为模型输入格式
# - Critic: 价值网络（baseline），用于估计每一步的状态价值
# - generate_samples / generate_experiences: 用当前策略生成样本并计算奖励、优势
# - ExperienceBuffer: 简单的经验缓存，用于微批量训练
# - train_step / train: 对策略网络（actor）和价值网络（critic）进行 PPO 更新
#
# 注：代码中使用了 "左填充" 的 tokenizer 设置，并且在生成时对序列进行截断/补齐。

# 构建dataset
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        
        """
        将一批原始提示（纯文本）转成 tokenizer 所需的最终 prompt 字符串列表。

        参数:
        - prompts: 原始文本提示列表
        - tokenizer: 对应的 tokenizer（需要能支持 apply_chat_template 可选接口）
        - apply_chat_template: 如果为 True，则按 chat 风格包装用户消息；否则在前面加 bos token

        最终生成的 `self.final_prompts` 存放的是已经经过模板处理或前缀添加的字符串，
        DataLoader 取样后直接传入 tokenizer 进行编码。
        """

        self.final_prompts = []

        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt

            self.final_prompts.append(prompt)
        
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.final_prompts[index]

# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        """
        价值网络（Critic）：
        使用已存在的 base_model（通常与 actor 的 backbone 相同）作为特征提取器，
        其输出接一个线性回归头 `value_head`，用于预测每个时间步的 state value。

        注意：`base_model` 在本实现中以 eval 模式加载（不更新 BN/dropout 等），
        价值网络的训练只会更新 `value_head`（以及如果需要可以解除 base_model 的冻结）。
        """
        
    def forward(self, input_ids, attention_mask, num_actions):
        """
        输入：
        - input_ids: 完整的 token id 序列，shape=(batch_size, seq_len)
        - attention_mask: 对应的 mask，shape=(batch_size, seq_len)
        - num_actions: 本次生成（响应）阶段的 token 数量（即需要返回的 value 序列长度）

        输出：
        - values: shape=(batch_size, num_actions)，对应最后 num_actions 个位置的状态价值

        说明：模型先计算 hidden states（shape=(batch_size, seq_len, hidden_size)），
        然后通过线性层得到每个位置的标量 value，接着取除了最后一步预测外的前 seq_len-1 个位置，
        最后只返回最后 `num_actions` 个时间步的值（生成的动作对应的位置）。
        """

        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        value_model_output = self.value_head(hidden_state)
        values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
        return values



def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    """
    计算 PPO 的策略损失（剪切版 TRPO surrogate objective）：

    输入：
    - log_probs: 当前策略在选择到动作时的 log-prob，shape=(batch, T)
    - old_log_probs: 参考（旧）策略的 log-prob，shape=(batch, T)
    - advantages: 每一步的优势估计 A_t，shape=(batch, T)
    - action_mask: 可选，对每个时间步是否为有效动作的 mask (0/1)，用于忽略填充或终止位置
    - clip_eps: PPO 剪切系数 eps

    计算步骤：
    - 1) ratio = exp(log πθ(a|s) - log π_old(a|s))
    - 2) surr = ratio * A, surr_clipped = clamp(ratio, 1-eps, 1+eps) * A
    - 3) loss = -mean(min(surr, surr_clipped))，对无效位置做 mask

    返回标量损失。
    """

    # 计算概率比率 r(θ) = exp(log πθ - log π_old)
    ratio = (log_probs - old_log_probs).exp()
    # 未裁剪的目标 surr1 = r * A
    surr1 = ratio * advantages
    # 裁剪比率后得到 surr2，用于控制更新幅度
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    # 对每个位置取更保守的目标（取最小值），并取负号作为要最小化的损失
    loss = -torch.min(surr1, surr2)
    # 如果没有 action_mask，先对时间维和批次维做平均
    if action_mask is None:
        return loss.mean(-1).mean()
    # 有 mask 时，先按时间维加和再归一化，再对 batch 平均
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    """
    计算价值网络的损失（均方误差），可选择使用 clipping 来稳定训练：

    - values: 当前价值预测，shape=(batch, T)
    - old_values: 旧的价值预测（用于 clipping），shape=(batch, T)
    - returns: 目标回报（returns），shape=(batch, T)
    - clip_eps: 如果提供则对 value 的变化进行截断，类似 PPO 中的 value clipping

    返回标量损失，支持 action_mask 来忽略填充位置。
    """

    # 如果使用 value clipping，则限制 values 相对于 old_values 的变化
    if clip_eps is not None:
        # values_clipped：将 value 的变化裁剪到 [-clip_eps, clip_eps]
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        # 两个候选 loss：裁剪后与裁剪前，取最大以防止 value 大幅漂移
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        # 普通 MSE
        loss = (values - returns) ** 2

    # 与 policy_loss 一样，支持 action_mask 对时间维归一化
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
        "seqs",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "num_actions"
    )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
          
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
        # 说明：append 接受一个 experiences 列表（每个元素是 Experience dataclass），
        # 将其中各字段按 batch 形式组织并追加到内部 buffer 中。buffer 类似一个简单的 FIFO 缓存，
        # 超过 limit 时会丢弃最早的元素。
        
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]
    

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


"""
Samples: 临时结构，用于保存一次 generate() 得到的原始张量数据。
- seqs: 整个序列（包含 prompt + 生成），shape=(batch, total_len)
- attention_mask: 对应 seqs 的 mask
- action_mask: 在生成部分哪些位置是有效动作（未到 eos 且非 pad）
- num_actions: 生成部分的长度（action_mask.size(1)）
- response_length/total_length: 记录每条样本的长度信息（用于后续裁剪/聚合）
"""

@dataclass
class Experience:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None


"""
Experience: 表示一次完整的经验条目，用于训练阶段抽取。
包含：
- seqs: 原始 token id 序列
- action_log_probs: 策略模型对生成 token 的 log-prob
- values: 价值网络预测
- returns: 计算得到的回报
- advantages: 计算得到的优势
- attention_mask / action_mask: 掩码信息
- reward: 奖励模型（或外部信号）给到的 reward
- response_length / total_length: 长度信息
- num_actions: 生成 token 数
- kl: 策略与参考策略间的近似 KL（用于计算奖励调整）
"""

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):

    """
    计算近似的 KL（在本实现中直接返回 log 比率）：

    - log_ratio = log π(a|s) - log π_ref(a|s)
    - 如果提供 action_mask，则忽略填充位置

    返回值是每一步的 log 比率（可以用来估计 KL 或作为调整奖励的项）。
    """

    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio

# A(t) = R(t) + gam*V(t+1) - V(t)
# gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
# returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float):
    
    """
    使用 GAE（Generalized Advantage Estimation）计算优势（advantages）和回报（returns）。

    公式回顾：
    - δ_t = r_t + γ * V_{t+1} - V_t
    - A_t = δ_t + γ * λ * A_{t+1}
    - returns(t) = A_t + V_t

    输入：
    - values: 价值网络对每个位置的估计，shape=(batch, T)
    - rewards: 每步的即时奖励，shape=(batch, T)
    - action_mask: 指示哪些位置是有效生成动作（用于屏蔽 pad/eos）
    - gamma, lambd: 折扣因子与 GAE 的 lambda

    返回：advantages（detach 后）与 returns
    """

    # 初始化 GAE 的累积变量
    lastgaelam = 0
    advantages_reversed = []
    # 响应长度（生成部分长度）
    response_length = rewards.size(1)

    # 屏蔽无效位置（pad 或 eos）以避免将它们纳入计算
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    # 逆向遍历时间步，逐步计算 δ_t 与 A_t
    for t in reversed(range(response_length)):
        # nextvalues：t+1 时刻的 value（如果 t 是最后一个位置，则为 0）
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        # δ_t = r_t + γ * V_{t+1} - V_t
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        # A_t = δ_t + γ * λ * A_{t+1}
        lastgaelam = delta + gamma * lambd * lastgaelam
        # 将计算到的 A_t 保存在列表（逆序）
        advantages_reversed.append(lastgaelam)
    # 反转回正序并堆叠为 (batch, T)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # returns = advantages + values
    returns = advantages + values
    # detach advantages：训练策略时不传播到 value 计算图
    return advantages.detach(), returns

def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], [])
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i:i+micro_rollout_batch_size]
        # 将 prompt 批次编码为 input_ids（左填充/定长）
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        # 使用模型的 generate 接口生成序列（包含 prompt + 生成）
        seqs = model.generate(**inputs.to(device), 
                            max_new_tokens = max_new_tokens, 
                            eos_token_id = eos_token_id, 
                            pad_token_id = pad_token_id)
        # 生成结果长度可能小于或大于目标长度，这里统一截断或用 pad 补齐到 max_new_tokens + max_length
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            # 在序列右侧补 pad，使所有样本具有相同的序列长度，方便后续张量运算
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)), fill_value=pad_token_id, device=seqs.device)], dim=1)

        # attention_mask 标识非 pad 的位置（1 表示有效 token）
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        # ans 为模型生成的部分（去除 prompt），其列数等于 max_new_tokens
        ans = seqs[:, input_ids.size(1):]
        # action_mask 标识生成部分哪些位置是真实动作（既不是 eos 也不是 pad），用于后续奖励/优势计算
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            # response_length: 每个样本生成部分有效 token 的个数（float tensor）
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):

        kl_divergence_estimate = -kl_ctl * kl
        rewards = kl_divergence_estimate

        ends = action_mask.sum(1) + 1
        
        if not isinstance(clip_reward_value, torch.Tensor):
            clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    
        reward_clip = torch.clamp(r, -clip_reward_value,
                                  clip_reward_value)
        batch_size = r.size(0)
        for j in range(batch_size):
            rewards[j, :ends[j]][-1] += reward_clip[j, 0]

        return rewards

# 说明：compute_rewards 将近似 KL（kl）和 reward model 的输出 r 组合成最终用于训练的奖励序列。
# - kl_ctl 用来缩放 KL 项，通常为正值，将产生负项降低生成不符合参考策略的行为。
# - reward_model 的输出会被裁剪并累加到每条样本的最后一个有效生成位置上（代表整条生成的任务得分）。

def generate_experiences(samples_list):

    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        with torch.no_grad():
            # 计算策略模型输出token的概率
            output = actor_model(seqs, attention_mask=attention_mask)
            logits = output.logits
            # logits: 模型在每个位置对 vocab 的未归一化分数，shape=(batch, seq_len, vocab_size)
            # 取 logits[:, :-1, :] 是因为第 i 个位置的 logits 对应预测下一个 token（i->i+1）
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            # log_probs: shape=(batch, seq_len-1, vocab_size)
            # 使用 gather 按照真实的下一个 token id（seqs[:,1:]）抽取对应位置的 log-prob
            # index 需要形状 (batch, seq_len-1, 1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            # squeeze 后为 (batch, seq_len-1)，取最后 num_actions 列对应的是生成（response）部分的 log-prob
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
            #计算参考模型输出token的概率
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 计算价值
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            # 转换成文本
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 计算奖励模型的奖励值
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            # 奖励模型通常对整段文本输出一个或多个标量（例如分类 logits），此处将 logits 作为 reward 信号
            # r 的 shape 依 reward_model 而异，常见为 (batch, 1)
            r = reward_model(**reward_model_inputs.to(device)).logits
            # 计算kl散度
            kl = compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask).to(device)
            # 计算实际奖励
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2)
            # 计算优势和回报
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)
        # actor_model.train()
        # critic_model.train()

        experiences.append(Experience(seqs,
                    action_log_probs.detach(),
                    value.detach(),
                    returns.detach(),
                    advantages.detach(),
                    attention_mask,
                    action_mask,
                    r.detach(),
                    samples.response_length,
                    samples.total_length,
                    num_actions,
                    kl.detach(),
        ))

    return experiences

@dataclass
class BufferItem:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    # 返回一个聚合后的 BufferItem，用于 DataLoader 的 batch 训练
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))
    
def train_step(experience, steps):
    
    actor_model.train()
    # 进入训练模式并清空 actor 的梯度
    optimizer_actor.zero_grad()

    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    logits = actor_model(
            sequences,
            attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
    # 说明：这里重新计算当前策略的 log-prob，目的是将最新策略与之前生成时记录的 old_action_log_probs 对比
    # 以计算 PPO 的 ratio，并最终得到 policy loss

    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages, action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()  
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")
    

def train():
    # 初始化经验池
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            # 生成样本（获取模型推理结果）
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            # 生成经验（获取优势、奖励、回报等）
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            
            buffer.clear()
        
            torch.cuda.empty_cache()


# train():
# - 主循环：对 prompts 数据集循环，每次从 prompts 中采样并用 actor 生成若干样本，
#   再用 reward_model/critic/ref_model 计算奖励和优势并放入 buffer，随后从 buffer 中构建 dataloader
#   对 actor/critic 进行若干 epoch 的更新。
# - 关键点：生成与训练分离（generate_experiences 在 no_grad 下完成），训练阶段重新计算当前策略的 logits
#   用于计算新的 log-prob 并与旧的 log-prob 比较以计算 PPO 损失。
            

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
    # 记录日志
    writer = SummaryWriter('./runs')
    # 策略模型
    #模型下载
    # model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',cache_dir="D:\Pretrained_models") 
    # actor_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    actor_model = AutoModelForCausalLM.from_pretrained("D:\Pretrained_models\Qwen/Qwen2.5-0.5B-Instruct").to(device)
    # 参考模型
    # ref_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    ref_model = AutoModelForCausalLM.from_pretrained("D:\Pretrained_models\Qwen/Qwen2.5-0.5B-Instruct").to(device)
    # 奖励模型
        #模型下载
    
    # model_dir = snapshot_download('deepset/deberta-v3-large-squad2',cache_dir = "D:\Pretrained_models")
    reward_model = AutoModelForSequenceClassification.from_pretrained("D:\Pretrained_models\deepset/deberta-v3-large-squad2").to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained("D:\Pretrained_models\Qwen/Qwen2.5-0.5B-Instruct")
    reward_tokenizer = AutoTokenizer.from_pretrained("D:\Pretrained_models\deepset/deberta-v3-large-squad2")
    # reward_model = AutoModelForSequenceClassification.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2').to(device)
    # actor_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    # 价值模型
    critic_model = Critic(actor_model.base_model).to(device)
    
    # 初始化优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
    
    # 填充方式为左填充
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
   
    train()
    

