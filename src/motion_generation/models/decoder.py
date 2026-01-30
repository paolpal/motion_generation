import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
from x_transformers import TransformerWrapper, Encoder, Decoder, CrossAttender
from x_transformers.x_transformers import TokenEmbedding, AttentionLayers

class XGestureDecoder(nn.Module):
    def __init__(
        self, 
        pose_vocab_size: int,
        transcript_vocab_size: Optional[int] = None,
        text_embedding: Optional[nn.Embedding] = None, # Il tuo embedding freezato
        d_model: Optional[int] = 300,
        nhead: int = 6,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        **kwargs
    ):
        super().__init__()
        assert text_embedding is not None or (transcript_vocab_size is not None and d_model is not None), \
            "Fornire text_embedding o la coppia transcript_vocab_size/d_model"

        if text_embedding is not None:
            if transcript_vocab_size is not None:
                assert text_embedding.num_embeddings == transcript_vocab_size, "Incoerenza vocab_size"
            if d_model is not None:
                assert text_embedding.embedding_dim == d_model, "Incoerenza d_model"
            
            # Estrazione valori effettivi
            d_model = text_embedding.embedding_dim
            transcript_vocab_size = text_embedding.num_embeddings

        # Ora d_model e transcript_vocab_size sono garantiti e coerenti
        assert d_model is not None, "d_model non può essere None a questo punto." # necessario per il type checker
        assert transcript_vocab_size is not None, "transcript_vocab_size non può essere None a questo punto."
        assert d_model % 2 == 0, f"d_model ({d_model}) deve essere pari per RoPE."

        self.config = {
            'transcript_vocab_size': transcript_vocab_size,
            'pose_vocab_size': pose_vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_seq_len': max_seq_len,
        }

        self.text_embedding = text_embedding if text_embedding is not None else nn.Embedding(transcript_vocab_size, d_model)
        self.pose_embedding = nn.Embedding(pose_vocab_size, d_model)

        self.decoder = Decoder(
                dim = d_model,
                depth = num_layers,
                heads = nhead,
                ff_mult = dim_feedforward // d_model,
                layer_dropout= dropout,
                attn_dropout = dropout,
                cross_attend = False,
                # cross_attn_tokens_dropout = dropout,
                rotary_pos_emb = True,  
                rotary_base_rescale_factor = 2
            )
        
        self.fc_out = nn.Linear(d_model, pose_vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, return_attn: bool=False):
        # 1. Embedding
        src = self.text_embedding(src) 
        tgt = self.pose_embedding(tgt) 

        # 2. Interleaving
        batch_size = src.shape[0]
        interleaved = torch.stack((src, tgt), dim=2).view(batch_size, -1, self.config['d_model'])

        # 3. Maschera
        mask = None
        if src_padding_mask is not None and tgt_padding_mask is not None:
            # Assumiamo maschere originali con True sui PAD: le invertiamo
            s_mask = ~src_padding_mask 
            t_mask = ~tgt_padding_mask
            mask = torch.stack((s_mask, t_mask), dim=2).view(batch_size, -1)

        # 4. Decoder
        # x-transformers: se return_hiddens=True, restituisce (output, hiddens)
        hiddens = None
        if return_attn:
            out, hiddens = self.decoder(interleaved, mask = mask, return_hiddens = True)
        else:
            out = self.decoder(interleaved, mask = mask)

        # 5. De-interlacciamento corretto
        # out è (B, 2*N, D)
        out_tgt = out[:, 1::2, :] # Prendi solo i token 'posa' (indici dispari)

        # 6. Head di uscita
        logits = self.fc_out(out_tgt)

        if return_attn:
            assert hiddens is not None, "hiddens non può essere None quando return_attn è True"
            attn_maps = [t.post_softmax_attn for t in hiddens.attn_intermediates]
            return logits, attn_maps
        
        return logits

    def save(self, path: Union[str, Path]):
        if isinstance(path, str): path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prendiamo lo state_dict completo
        full_state_dict = self.state_dict()
        
        # Rimuoviamo tutte le chiavi che appartengono all'embedding del testo
        # In x-transformers, l'embedding dell'encoder è sotto 'encoder.token_emb'
        keys_to_exclude = [k for k in full_state_dict.keys() if 'text_embedding' in k]
        
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if k not in keys_to_exclude}

        torch.save({
            'model_state_dict': filtered_state_dict,
            'config': self.config
        }, path)
        print(f"Modello (senza Text Embedding) salvato in {path}")

    @classmethod
    def load(cls, path: Union[str, Path], text_embedding: nn.Embedding, device: Union[str, torch.device]='cpu'):
        checkpoint = torch.load(path, map_location=device)
        
        # Inizializziamo la classe: passerà l'embedding pesante appena fornito
        # all'interno di x-transformers
        model = cls(text_embedding=text_embedding, **checkpoint['config'])
        
        # Carichiamo i pesi. Usiamo strict=False perché nel file salvato 
        # mancano i pesi dell'embedding che abbiamo rimosso nel save()
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Missing keys (Expected: text_embedding): {incompatible_keys.missing_keys}")
        
        model.to(device)
        model.eval()
        return model
