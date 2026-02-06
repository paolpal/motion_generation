from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Union, Dict

class GestureRNN(nn.Module):
    def __init__(self, 
            pose_vocab_size: int,
            transcript_vocab_size: Optional[int] = None,
            text_embedding: Optional[nn.Embedding] = None, # Il tuo embedding freezato
            d_model: Optional[int] = 300,
            num_layers: int = 1,
            dropout: float = 0.1):
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

        self.config = {
            'transcript_vocab_size': transcript_vocab_size,
            'pose_vocab_size': pose_vocab_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'dropout': dropout,
        }

        self.embed_text = nn.Embedding(transcript_vocab_size, d_model) if text_embedding is None else text_embedding
        self.embed_pose = nn.Embedding(pose_vocab_size, d_model)
        
        # Encoder: Gru o LSTM
        self.encoder = nn.GRU(d_model, d_model, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        # Decoder: riceve l'output dell'embedding + l'hidden state dell'encoder
        self.decoder = nn.GRU(d_model, d_model, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(d_model, pose_vocab_size)

    def forward(self, src, tgt):
        # 1. Encoding
        text_emb = self.embed_text(src)
        _, hidden = self.encoder(text_emb) # hidden è il nostro "riassunto" del testo
        
        # 2. Decoding
        pose_emb = self.embed_pose(tgt)
        # Passiamo l'hidden state dell'encoder come stato iniziale del decoder
        output, _ = self.decoder(pose_emb, hidden)
        
        return self.fc_out(output)
    
    def save(self, path: Union[str, Path], save_text_embedding: bool = False):
        if isinstance(path, str): path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prendiamo lo state_dict completo
        full_state_dict = self.state_dict()
        
        # Rimuoviamo tutte le chiavi che appartengono all'embedding del testo
        if not save_text_embedding:
            keys_to_exclude = [k for k in full_state_dict.keys() if 'embed_text' in k]
        else:
            keys_to_exclude = []
        
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if k not in keys_to_exclude}

        torch.save({
            'model_state_dict': filtered_state_dict,
            'config': self.config
        }, path)

        if not save_text_embedding:
            print(f"Modello salvato senza embedding del testo in {path}")
        else:
            print(f"Modello salvato con embedding del testo in {path}")

    @classmethod
    def load(cls, path: Union[str, Path], text_embedding: nn.Embedding, device: Union[str, torch.device]='cpu'):
        checkpoint = torch.load(path, map_location=device)
        
        # Inizializziamo la classe: passerà l'embedding pesante appena fornito
        # all'interno di x-transformers
        model = cls(text_embedding=text_embedding, **checkpoint['config'])
        
        # Carichiamo i pesi. Usiamo strict=False perché nel file salvato 
        # mancano i pesi dell'embedding che abbiamo rimosso nel save()
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Missing keys (Expected: embed_text): {incompatible_keys.missing_keys}")
        
        model.to(device)
        model.eval()
        return model