from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
import math

from motion_generation.embedding import CompositeEmbedding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = self.get_buffer('pe')
        x = x + pe[:x.size(0)]
        return self.dropout(x)

class GestureTransformer(nn.Module):
    def __init__(
        self, 
        transcript_vocab_size: int,
        pose_vocab_size: int,
        text_embedding: Optional[nn.Embedding] = None,
        d_model: int = 300,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        look_ahead_steps: int = 3,
        context_back_steps: Optional[int] = None,
        max_seq_len: int = 500,
        memory_causal: bool = False
    ):
        super().__init__()

        if text_embedding is not None:
            assert text_embedding.embedding_dim == d_model, \
                "The Embedding for text must match the model's d_model."
            assert text_embedding.num_embeddings == transcript_vocab_size, \
                "The Embedding for text must match the transcript_vocab_size."
            
        assert look_ahead_steps >= 0, "look_ahead_steps must be non-negative"
        if context_back_steps is not None:
            assert context_back_steps >= 0, "context_back_steps must be non-negative"
        
        # Salva la config per save/load
        self.config = {
            'transcript_vocab_size': transcript_vocab_size,
            'pose_vocab_size': pose_vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'look_ahead_steps': look_ahead_steps,
            'context_back_steps': context_back_steps,
            'max_seq_len': max_seq_len,
            'memory_causal': memory_causal,
        }

        self.look_ahead_steps = look_ahead_steps
        self.context_back_steps = context_back_steps
        
        # 1. Embedding & Proiezioni
        self.text_embedding = text_embedding if text_embedding is not None else nn.Embedding(transcript_vocab_size, d_model)
        self.pose_embedding = nn.Embedding(pose_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 2. Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Output Head
        self.fc_out = nn.Linear(d_model, pose_vocab_size)

    def get_look_ahead_causal_mask(self, tgt_len, src_len=None):
        # Creiamo una maschera dove tutto è inizialmente visibile (0)
        # Una maschera di True/False dove True significa "NASCONDI"
        
        # 1. Prendiamo la parte superiore oltre il look_ahead_steps
        # diagonal=1 è la diagonale subito sopra la principale
        if src_len is None:
            src_len = tgt_len
        mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=self.look_ahead_steps + 1).bool()
    
        return mask
    
    def get_causal_mask(self, seq_len):
        # Maschera triangolare superiore standard
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    def get_sliding_window_mask(self, tgt_len, src_len=None):
        if src_len is None:
            src_len = tgt_len

        mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=self.look_ahead_steps + 1).bool()
        if self.context_back_steps is not None:
            lower_mask = torch.tril(torch.ones(tgt_len, src_len), diagonal=-self.context_back_steps - 1).bool()
            mask = mask | lower_mask

        return mask

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # src: [batch, seq_len_text]
        # tgt: [batch, seq_len_pose] (indici)
        
        src = self.text_embedding(src)
        src = self.pos_encoder(src)
        
        tgt_emb = self.pose_embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Maschera causale per il decoder (fondamentale in train)
        tgt_mask = self.get_causal_mask(tgt.size(1)).to(tgt.device)
        src_mask = self.get_look_ahead_causal_mask(src.size(1)).to(src.device)
        if self.config['memory_causal']:
            memory_mask = self.get_look_ahead_causal_mask(tgt.size(1), src.size(1)).to(src.device)
            memory_padding_mask = src_padding_mask
        else:
            memory_mask = None
            memory_padding_mask = None
        out = self.transformer(
            src, tgt_emb, 
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
            memory_mask=memory_mask
        )
        
        return self.fc_out(out)

    def save(self, path: Union[str, Path], save_text_embedding: bool = False):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.state_dict()

        if not save_text_embedding:
            keys_to_remove = [k for k in state_dict.keys() if k.startswith('text_embedding')]
            for k in keys_to_remove:
                del state_dict[k]

        torch.save({
            'model_state_dict': state_dict,
            'config': self.config,
            'save_text_embedding': save_text_embedding
        }, path)
        print(f"Modello salvato in {path}")

    @classmethod
    def load(cls, path: Union[str, Path], text_embedding: Optional[nn.Embedding] = None, device:Union[str, torch.device]='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        save_text_embedding = checkpoint.get('save_text_embedding', False)
        if not save_text_embedding and text_embedding is None:
            raise ValueError("Il modello è stato salvato senza l'embedding del testo. Devi fornire un'embedding di testo valido per caricare il modello.")
        model = cls(text_embedding=text_embedding, **config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        return model