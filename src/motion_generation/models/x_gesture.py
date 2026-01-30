import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
from x_transformers import TransformerWrapper, Encoder, Decoder, CrossAttender
from x_transformers.x_transformers import TokenEmbedding, AttentionLayers

class XGestureTransformer(nn.Module):
    def __init__(
        self, 
        pose_vocab_size: int,
        transcript_vocab_size: Optional[int] = None,
        text_embedding: Optional[nn.Embedding] = None, # Il tuo embedding freezato
        d_model: Optional[int] = 300,
        nhead: int = 6,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        look_ahead_steps: int = 9,      # ~0.6s @ 15FPS (Anticipo biologico)
        context_back_steps: Optional[int] = 18,   # ~1.2s @ 15FPS (Durata gesto)
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
        assert d_model % 2 == 0, f"d_model ({d_model}) deve essere pari per RoPE."

        # Salvataggio configurazione
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
        }

        self.look_ahead_steps = look_ahead_steps
        self.context_back_steps = context_back_steps


        if text_embedding is not None:
            text_embedding_wrapper = TokenEmbedding(
                num_tokens=transcript_vocab_size,
                dim=d_model
            )
            text_embedding_wrapper.emb = text_embedding 
        else:
            text_embedding_wrapper = None  # Usa l'embedding interno di x-transformers

        # 1. ENCODER (Testo)
        # Riceve i token, usa il tuo Word2Vec e applica RoPE
        
        self.encoder = TransformerWrapper(
            num_tokens = transcript_vocab_size,
            max_seq_len = max_seq_len,
            token_emb = text_embedding_wrapper,
            tie_embedding=True,
            attn_layers = AttentionLayers(
                dim = d_model,
                depth = num_encoder_layers,
                heads = nhead,
                ff_mult = dim_feedforward // d_model,
                attn_dropout = dropout,
                layer_dropout= dropout,
                causal = True,
                rotary_pos_emb = True
            )
        )

        # 2. DECODER (Pose)
        # Genera pose in modo autoregressivo
        self.decoder = TransformerWrapper(
            num_tokens = pose_vocab_size,
            max_seq_len = max_seq_len,
            attn_layers = AttentionLayers(
                dim = d_model,
                depth = num_decoder_layers,
                heads = nhead,
                ff_mult = dim_feedforward // d_model,
                layer_dropout= dropout,
                attn_dropout = dropout,
                causal = True,
                cross_attend = True,
                # cross_attn_tokens_dropout = dropout,
                rotary_pos_emb = True,  
                rotary_base_rescale_factor = 2
            )
        )

    def get_gestural_mask(self, tgt_len, src_len, device):
        # Assicuriamoci che rows sia la dimensione verticale (Tgt) 
        # e cols quella orizzontale (Src)
        rows = torch.arange(tgt_len, device=device).view(-1, 1) # (tgt, 1)
        cols = torch.arange(src_len, device=device).view(1, -1) # (1, src)

        future_mask = cols > (rows + self.look_ahead_steps)
        
        if self.context_back_steps is not None:
            past_mask = cols < (rows - self.context_back_steps)
        else:
            past_mask = torch.zeros((tgt_len, src_len), device=device, dtype=torch.bool)

        mask = future_mask | past_mask
        
        # Restituiamo (1, 1, tgt_len, src_len)
        return ~mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, return_attn=False):

        if src_padding_mask is not None:
            src_padding_mask = ~src_padding_mask  # Inverti per x-transformers
        if tgt_padding_mask is not None:
            tgt_padding_mask = ~tgt_padding_mask  # Inverti per x-transformers

        # 1. Encoder causale
        enc_out = self.encoder(
            src, 
            mask=src_padding_mask, 
            return_embeddings=True
        )

        # 2. Maschera asimmetrica biologica
        cross_attn_mask = self.get_gestural_mask(tgt.size(1), src.size(1), src.device)

        # 3. Decoding con cattura delle attenzioni
        # Usiamo le keyword di x-transformers per estrarre i pesi
        if return_attn:
            # Ritorna (logits, intermediate_attn_data)
            logits, intermediate = self.decoder(
                tgt, 
                context=enc_out, 
                mask=tgt_padding_mask,
                context_mask=src_padding_mask,
                cross_attn_mask=cross_attn_mask,
                return_attn=True # MHSA = Multi-Head Self Attention + Cross
            )
            
            # intermediate è una lista di dizionari (uno per layer)
            # Cerchiamo la 'cross_attn' in ogni layer
            
            return logits, intermediate
        
        # Forward standard
        return self.decoder(
            tgt, 
            context=enc_out, 
            context_mask=src_padding_mask,
            mask=tgt_padding_mask,
            cross_attn_mask=cross_attn_mask,
        )


    def save(self, path: Union[str, Path]):
        if isinstance(path, str): path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prendiamo lo state_dict completo
        full_state_dict = self.state_dict()
        
        # Rimuoviamo tutte le chiavi che appartengono all'embedding del testo
        # In x-transformers, l'embedding dell'encoder è sotto 'encoder.token_emb'
        keys_to_exclude = [k for k in full_state_dict.keys() if 'encoder.token_emb' in k]
        
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
        print(f"Missing keys (Expected: token_emb): {incompatible_keys.missing_keys}")
        
        model.to(device)
        model.eval()
        return model

