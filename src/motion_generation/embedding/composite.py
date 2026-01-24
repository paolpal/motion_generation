import torch
import torch.nn as nn

class CompositeEmbedding(nn.Module):
    def __init__(
        self,
        word_embedding: nn.Embedding,
        d_model: int,
        bos_id: int,
        eos_id: int,
    ):
        super().__init__()
        
        self.word_embedding = word_embedding
        self.word_embedding.weight.requires_grad = False

        self.bos_id = bos_id
        self.eos_id = eos_id

        self.special_embedding = nn.Embedding(2, d_model)

    def forward(self, input_ids):
        """
        input_ids: [B, T]
        """
        B, T = input_ids.shape
        D = self.special_embedding.embedding_dim
        
        out = torch.zeros(B, T, D, device=input_ids.device)

        # BOS
        bos_mask = input_ids == self.bos_id
        if bos_mask.any():
            out[bos_mask] = self.special_embedding(
                torch.zeros_like(input_ids[bos_mask])
            )

        # EOS
        eos_mask = input_ids == self.eos_id
        if eos_mask.any():
            out[eos_mask] = self.special_embedding(
                torch.ones_like(input_ids[eos_mask])
            )

        # parole normali (+ UNK)
        normal_mask = ~(bos_mask | eos_mask)
        if normal_mask.any():
            out[normal_mask] = self.word_embedding(input_ids[normal_mask])

        return out
