import torch
import torch.nn as nn

class MeanPoolTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, L), attention_mask: (B, L)
        x = self.emb(input_ids)  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, H)
        return self.ln(x)