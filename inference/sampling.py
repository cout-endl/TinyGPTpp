import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    从 logits 中采样下一个 token，支持 temperature、Top‑k、Top‑p（核采样）。
    若 temperature==0，直接返回 argmax（贪心）。
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 温度缩放
    logits = logits / temperature

    vocab_size = logits.size(-1)

    # ---- Top-k ----
    if top_k is not None:
        if top_k > vocab_size:
            top_k = vocab_size
        topk_values, _ = torch.topk(logits, top_k, dim=-1)
        min_topk = topk_values[:, -1].unsqueeze(-1)          # (B, 1)
        logits = torch.where(
            logits < min_topk,
            torch.tensor(float('-inf'), device=logits.device),
            logits
        )

    # ---- Top-p (核采样) ----
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 生成排序后的掩码：累积概率 > top_p 的位置为 True
        sorted_mask = cum_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()   # 保留第一个超过的 token
        sorted_mask[:, 0] = 0

        # 将排序后的掩码映射回原始位置
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_mask)      # scatter_ 将 sorted_mask 按索引填入 mask

        # 将需要屏蔽的 token 设为 -inf
        logits.masked_fill_(mask, float('-inf'))

    # ---- 防止全被屏蔽导致 multinomial 报错 ----
    if torch.all(logits == float('-inf')):
        # 所有词都被屏蔽 → 退化为均匀分布
        logits = torch.ones_like(logits) / vocab_size
    else:
        logits = F.softmax(logits, dim=-1)

    next_token = torch.multinomial(logits, num_samples=1)
    return next_token