import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_p=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    for _ in range(max_new_tokens):
        # 1. 裁剪序列长度
        # 如果当前的序列长度超过了模型的最大上下文长度 (block_size)，需要截断
        # 假设 model 有一个 config 属性存储了 block_size
        idx_cond = idx if idx.size(1) <= model.config.context_length else idx[:, -model.config.context_length:]
        
        # 2. 前向传播
        # 获取 logits，形状通常是 (batch, seq_len, vocab_size)
        logits, _ = model(idx_cond)
        
        # 3. 只需要最后一个时间步的 logits
        # 形状变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        
        # 4. Apply Temperature (温度缩放)
        # 温度越高，分布越平（越随机）；温度越低，分布越尖（越保守）
        logits = logits / temperature
        
        # 5. Apply Top-p (Nucleus) Sampling
        if top_p is not None and top_p < 1.0:
            # 5.1 排序：从大到小排序 logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # 5.2 计算 softmax 后的累积概率 (Cumulative Probabilities)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 5.3 创建移除掩码 (Mask)
            # 我们要移除那些累计概率超过 top_p 的词
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # [关键细节]：Shift mask right
            # 假设 top_p=0.9, 累积概率是 [0.8, 0.89, 0.95, ...]
            # 原始 mask 是 [False, False, True, ...]，这意味着第三个词就被移除了。
            # 但实际上我们需要保留第一个让概率超过 0.9 的词（否则可能连一个词都选不出来）。
            # 所以我们将 mask 向右移动一位：
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0 # 确保第一个词永远被保留
            
            # 5.4 将 mask 映射回原始的 logits 索引
            # 使用 scatter 或者直接用 gather 出来的索引去修改原 logits 比较麻烦
            # 通常做法是：直接修改 sorted_logits，然后再映射回去，或者直接在 sorted 空间采样
            # 这里我们为了通用性，把 mask 映射回原始位置并填入 -inf
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # 6. 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        
        # 7. 从多项分布中采样 1 个 token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 8. 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)
        
        # (可选) 如果遇到 <EOS> token 可以提前 break，作业视要求而定
        if idx == model.config.eos_token:
            break
        
    return idx