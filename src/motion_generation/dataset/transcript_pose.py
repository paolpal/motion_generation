import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from pats.utils import Skeleton2D
from motion_quantization.quantization import PoseTokenizer
from motion_generation.tokenizer import Word2VecTokenizer

from pats.utils import load_multiple_samples, get_speaker_intervals
from typing import List, Optional, Tuple, Union, Literal, Dict


class TranscriptPoseDataset(Dataset):
    def __init__(self, 
                 speakers: List[str],
                 data_root: Union[str, Path],
                 pose_tokenizer: PoseTokenizer, 
                 text_tokenizer: Word2VecTokenizer, 
                 split: Literal["train", "dev", "test"] = "train",
                 cache_path: Optional[Union[str, Path]] = None):
        """
        Dataset con supporto per caching automatico.
        """
        self.speakers = sorted(speakers)
        self.data_root = Path(data_root)
        self.pose_tokenizer = pose_tokenizer
        self.text_tokenizer = text_tokenizer
        
        # 1. Tentativo di caricamento dalla cache
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f"--- Caricamento cache da {cache_path} ---")
                checkpoint = torch.load(cache_path)
                self.data = checkpoint['data']
                # Opzionale: verifica se gli speaker nel file coincidono con quelli richiesti
                return # Esci dal costruttore se caricamento completato

        # 2. Se la cache non esiste o non è stata passata, esegui il processing
        print(f"--- Cache non trovata. Avvio processing per {split} ---")
        samples = self._load_raw_dataset_parallel(split)
        self.data = self._build_tokenized_samples(samples)

        # 3. Salvataggio automatico se richiesto
        if cache_path is not None:
            self.save(cache_path)

    @staticmethod
    def _load_raw_speaker_samples(speaker: str, split: str, data_root: Path) -> List[Dict]:
        """Eseguito in processi separati per velocizzare l'I/O e la normalizzazione."""
        intervals = get_speaker_intervals(speaker=speaker, split=split, data_root=data_root)
        raw_samples = load_multiple_samples(speaker=speaker, interval_ids=intervals, data_root=data_root)

        samples: List[Dict] = []
        for s in raw_samples:
            pose = s['pose']
            # Centering e Normalizzazione
            pose[:, 0] = [0.0, 0.0] 
            pose = Skeleton2D.normalize_skeleton(pose)
            
            samples.append({
                'text': s['text'],
                'words': s['words'],
                'n_frames': s['n_frames'],
                'pose': pose # Restituisce numpy per evitare problemi di serializzazione
            })
        return samples

    @torch.no_grad()
    def _load_raw_dataset_parallel(self, split: str) -> List[Dict]:
        print(f"--- Caricamento {split} set in corso (Parallelized) ---")
        all_samples: List[Dict] = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._load_raw_speaker_samples, s, split, self.data_root) for s in self.speakers]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading {split}"):
                try:
                    all_samples.extend(future.result())
                except Exception as e:
                    print(f"Errore caricamento speaker: {e}")
        return all_samples

    def __len__(self) -> int:
        return len(self.data)
    
    def encode_text_framewise(self, words: List[Dict], length: int) -> torch.Tensor:
        """
        Tokenizza il testo frame-aligned in embedding densi.
        
        Args:
            words: Lista di dizionari con 'word', 'start', 'end'.
            length: Numero totale di frame per l'allineamento.
        Returns:
            Tensor di shape (T, 300) con gli embedding word2vec.
        """
        text_tokens = torch.tensor([self.text_tokenizer.sil_id] * length, dtype=torch.long)  # Inizializza con SIL
        for w in words:
            text_tokens[int(w['start']):int(w['end'])] = self.text_tokenizer.encode(w['word'])

        return text_tokens
    
    def _build_tokenized_samples(self, samples: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        """
        Processa una lista di campioni: tokenizza il testo e quantizza la pose.
        
        Args:
            samples: Lista di dizionari con 'words', 'n_frames', 'pose'.
        Returns:
            Dizionario con 'text_tokens' e 'motion_tokens'.
        """
        processed_samples = []
        for idx, sample in enumerate(tqdm(samples, desc="Processing Samples")):
            # 1. Tokenizzazione Testo (senza padding, lo fa il collate)
            text_tokens = self.encode_text_framewise(sample['words'], sample['n_frames'])

            # 2. Quantizzazione Pose -> Indices
            pose_tensor = torch.tensor(sample['pose'], dtype=torch.float32)
            pose_tensor = pose_tensor.reshape(-1, 104)  
            # Il quantizer si aspetta (B, T, J, C)
            _, indices = self.pose_tokenizer.quantize(pose_tensor)
            indices = indices.cpu() # (T_quantized)

            # 3. Preparazione Sequenza Finale: [BOS] + [Tokens + OFFSET] + [EOS]
            motion_tokens = self.pose_tokenizer.add_special_tokens(indices.unsqueeze(0), add_som=True, add_eom=True).squeeze(0)
            processed_samples.append({
                'text_tokens': text_tokens,
                'motion_tokens': motion_tokens
            })
        return processed_samples
        

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        return sample['text_tokens'], sample['motion_tokens']


    def save(self, save_path: Union[str, Path]):
        """
        Salva i dati processati e i metadati in un file .pt.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_to_save = {
            'data': self.data,
            'metadata': {
                'speakers': self.speakers,
                'data_root': str(self.data_root),
                # Salviamo i parametri del tokenizer se necessario, 
                # o almeno la loro configurazione
            }
        }
        
        print(f"--- Salvataggio dataset in {save_path} ---")
        torch.save(data_to_save, save_path)
        print("Salvataggio completato.")

    @classmethod
    def load(cls, 
             load_path: Union[str, Path], 
             pose_tokenizer: PoseTokenizer, 
             text_tokenizer: Word2VecTokenizer) -> 'TranscriptPoseDataset':
        """
        Carica un dataset pre-processato senza rieseguire la tokenizzazione.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Nessun file trovato in {load_path}")

        print(f"--- Caricamento dataset da {load_path} ---")
        checkpoint = torch.load(load_path)
        
        # Creiamo un'istanza "vuota" bypassando l'init pesante
        # __new__ alloca l'oggetto senza chiamare __init__
        instance = cls.__new__(cls)
        
        # Ripristiniamo gli attributi dai metadati
        instance.speakers = checkpoint['metadata']['speakers']
        instance.data_root = Path(checkpoint['metadata']['data_root'])
        instance.pose_tokenizer = pose_tokenizer
        instance.text_tokenizer = text_tokenizer
        
        # Carichiamo i dati processati
        instance.data = checkpoint['data']
        
        print(f"Dataset caricato con successo: {len(instance.data)} campioni.")
        return instance