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
    @torch.no_grad()
    def __init__(self, 
                 speakers: List[str],
                 data_root: Union[str, Path],
                 pose_tokenizer: PoseTokenizer, 
                 text_tokenizer: Word2VecTokenizer, 
                 max_len: Optional[int] = 400,
                 overlap: int = 100,
                 split: Literal["train", "dev", "test"] = "train",
                 cache_path: Optional[Union[str, Path]] = None,
                 force_rebuild: bool = False):
        """
        Dataset con supporto per caching automatico.
        """
        self.speakers = sorted(speakers)
        self.data_root = Path(data_root)
        self.pose_tokenizer = pose_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_len = max_len
        self.overlap = overlap

        pose_tokenizer.model.eval()
        pose_tokenizer.codebook.detach()

        # 1. Tentativo di caricamento dalla cache
        if cache_path is not None:
            if isinstance(cache_path, str):
                cache_path = Path(cache_path)
            spk_hash = "_".join(self.speakers[:3]) + f"_etc_{len(self.speakers)}"
            self.cache_file = cache_path / f"{spk_hash}_{split}.pt"
        else:
            self.cache_file = None


        if (
            self.cache_file is not None and
            self.cache_file.exists() and
            not force_rebuild
        ):
            print(f"Caricamento cache: {self.cache_file}")
            self.data = torch.load(self.cache_file, weights_only=False)
        else:
        
            print(f"--- Cache non trovata. Avvio processing per {split} ---")
            samples = self._load_raw_dataset_parallel(split)
            self.data = self._build_tokenized_samples(samples)
            if self.max_len is not None:
                self.data = self._build_windowed_samples(self.data)

            self.data = self._add_special_tokens(self.data)
            

        # 3. Salvataggio automatico se richiesto
        if self.cache_file is not None and (not self.cache_file.exists() or force_rebuild):
            self.save(self.cache_file)
    
    @staticmethod
    def _load_sample(speaker: str, interval_id: str, data_root: Path) -> Optional[Dict]:
        """Carica un singolo campione (usato per il caricamento parallelo)."""
        try:
            sample = load_multiple_samples(speaker=speaker, interval_ids=[interval_id], data_root=data_root)[0]
        except Exception as e:
            return None
        pose = sample['pose']
        # Centering e Normalizzazione
        pose[:, 0] = [0.0, 0.0] 
        pose = Skeleton2D.normalize_skeleton(pose)
        
        return {
            'text': sample['text'],
            'words': sample['words'],
            'n_frames': sample['n_frames'],
            'pose': pose # Restituisce numpy per evitare problemi di serializzazione
        }

    @torch.no_grad()
    def _load_raw_dataset_parallel(self, split: str) -> List[Dict]:
        print(f"--- Caricamento {split} set in corso (Parallelized) ---")
        speaker_intervals : List[Dict[str, str]] = []
        for s in tqdm(self.speakers, desc=f"Loading {split}"):
            intervals = get_speaker_intervals(speaker=s, split=split, data_root=self.data_root)
            for interval_id in intervals:
                speaker_intervals.append({
                    'speaker': s,
                    'sample_id': interval_id
                })
        all_samples: List[Dict] = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(TranscriptPoseDataset._load_sample, s['speaker'], s['sample_id'], self.data_root) for s in speaker_intervals]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading {split}"):
                try:
                    sample = future.result()
                    if sample is not None:
                        all_samples.append(sample)
                except Exception as e:
                    print(f"Errore caricamento speaker: {e}")
        return all_samples

    def __len__(self) -> int:
        return len(self.data)
    
    @torch.no_grad()
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

        return text_tokens.detach()
    
    @torch.no_grad()
    def _build_tokenized_samples(self, samples: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processa una lista di campioni: tokenizza il testo e quantizza la pose.
        
        Args:
            samples: Lista di dizionari con 'words', 'n_frames', 'pose'.
        Returns:
            Lista di tuple (text_tokens, motion_tokens).
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
            indices = indices.detach().cpu() # (T_quantized)

            # 3. Preparazione Sequenza Finale: [BOS] + [Tokens + OFFSET] + [EOS]
            motion_tokens = indices # self.pose_tokenizer.add_special_tokens(indices.unsqueeze(0), add_som=True, add_eom=False).squeeze(0)
            processed_samples.append((text_tokens, motion_tokens))
        return processed_samples
    
    def _add_special_tokens(self, data : List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        processed_data : List[Tuple[torch.Tensor, torch.Tensor]] = []
        for sample in data:
            text_tokens, motion_tokens = sample
            # Aggiungi special tokens
            motion_tokens = self.pose_tokenizer.add_special_tokens(motion_tokens.unsqueeze(0), add_som=True, add_eom=False).squeeze(0)   
            processed_data.append((text_tokens, motion_tokens))
        return processed_data

    @torch.no_grad()
    def _build_windowed_samples(self, samples) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        assert self.max_len is not None, "max_len deve essere specificato per il windowing."
        max_len = self.max_len
        overlap = self.overlap
        step = self.max_len - overlap
        windowed_samples = []
        for sample in tqdm(samples, desc="Building Windowed Samples"):
            full_text_tokens, full_motion_tokens = sample
        
            # Se la sequenza è già corta, la teniamo così com'è
            if len(full_text_tokens) <= max_len:
                windowed_samples.append((full_text_tokens, full_motion_tokens))
            else:
        
                # Se è lunga, facciamo sliding window
                for i in range(0, len(full_text_tokens) - max_len + 1, step):
                    t_chunk = full_text_tokens[i : i + max_len]
                    m_chunk = full_motion_tokens[i : i + max_len]
                    windowed_samples.append((t_chunk, m_chunk))
                
                # Gestiamo l'ultimo pezzo se è rimasto fuori (opzionale)
                if len(full_text_tokens) % step != 0:
                    t_chunk = full_text_tokens[-max_len:]
                    m_chunk = full_motion_tokens[-max_len:]
                    windowed_samples.append((t_chunk, m_chunk))
            
        return windowed_samples
        

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    
    def save(self, save_path: Union[str, Path]):
        """
        Salva i dati processati e i metadati in un file .pt.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"--- Salvataggio dataset in {save_path} ---")
        torch.save(self.data, save_path)
        print("Salvataggio completato.")
    

def transcript_motion_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function per batching di (text_tokens, motion_tokens).
    Esegue il padding dinamico su entrambe le sequenze.

    Args:
        batch: lista di tuple (text_tokens, motion_tokens), 
               text_tokens: Tensor[T_text], motion_tokens: Tensor[T_motion]

    Returns:
        Dict con:
            - 'text_tokens': Tensor[B, T_text_max]
            - 'motion_tokens': Tensor[B, T_motion_max]
            - 'text_mask': Tensor[B, T_text_max] (1 = token reale, 0 = PAD)
            - 'motion_mask': Tensor[B, T_motion_max] (1 = token reale, 0 = PAD)
    """
    text_seqs, motion_seqs = zip(*batch)  # unzip
    text_seqs = list(text_seqs)
    motion_seqs = list(motion_seqs)

    # Padding dinamico per testo e motion
    padded_texts = pad_sequence(text_seqs, batch_first=True, padding_value=0)    # PAD = 0
    padded_motions = pad_sequence(motion_seqs, batch_first=True, padding_value=0) # PAD = 0

    # Maschere (1 = token reale, 0 = PAD)
    text_mask = (padded_texts == 0)
    motion_mask = (padded_motions == 0)

    return {
        'text_tokens': padded_texts,       # [B, T_text_max]
        'motion_tokens': padded_motions,   # [B, T_motion_max]
        'text_mask': text_mask,            # [B, T_text_max]
        'motion_mask': motion_mask         # [B, T_motion_max]
    }