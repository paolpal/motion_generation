import os

import wandb

from motion_generation.embedding import Word2VecWeightFactory
os.environ["GENSIM_DATA_DIR"] = "/home/ubuntu/palumbo/Posemi/dataset/gensim_data"
import gensim.downloader as api
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from pathlib import Path
from tqdm import tqdm

from motion_generation.models import XGestureTransformer
from motion_generation.dataset import TranscriptPoseDataset, transcript_motion_collate_fn


from motion_quantization.models import SkeletonVQVAE
from motion_quantization.quantization import PoseTokenizer
from motion_generation.tokenizer import Word2VecTokenizer

PRJ_ROOT = Path("/home/ubuntu/palumbo/Posemi/")
WANDB_API_KEY = "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"

def train_one_epoch(model, dataloader, optimizer: Optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        encoder_input = batch['text_tokens'].to(device)           # (B, T, 300)
        decoder_input = batch['motion_tokens'][:, :-1].to(device)           # (B, M)
        decoder_target = batch['motion_tokens'][:, 1:].to(device)         # (B, M)
        encoder_padding_mask = batch['text_mask'].to(device)
        decoder_padding_mask = batch['motion_mask'][:, 1:].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            src=encoder_input,
            tgt=decoder_input,
            src_padding_mask=encoder_padding_mask,
            tgt_padding_mask=decoder_padding_mask
        )  # (B, M, vocab_size)
        
        # Calcola loss ignorando i token di padding (PAD_TOKEN=0)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # (B*M, vocab_size)
            decoder_target.reshape(-1)             # (B*M)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Validation"):
        encoder_input = batch['text_tokens'].to(device)
        decoder_input = batch['motion_tokens'][:, :-1].to(device)
        decoder_target = batch['motion_tokens'][:, 1:].to(device)
        encoder_padding_mask = batch['text_mask'].to(device)
        decoder_padding_mask = batch['motion_mask'][:, 1:].to(device)
        
        logits = model(
            src=encoder_input,
            tgt=decoder_input,
            src_padding_mask=encoder_padding_mask,
            tgt_padding_mask=decoder_padding_mask
        )
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1)
        )
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(config):

    hyperparams = {
        'max_seq_len': 500,
        'batch_size': 32,
        'nhead': 6,
        'encoder_layers': 2,
        'decoder_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.2,
        'lr': 1e-6,
        'weight_decay': 1e-5,
        'look_ahead_steps': 9,
        'context_back_steps': 45,
    }

    run = wandb.init(
        project="gesture_transformer", 
        config={**config, **hyperparams}, 
    )

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== 1. Carica Pose e Text Tokenizer e Text Embedding =====

    vqvae = SkeletonVQVAE.load(config['pose_tokenizer_path'])
    pose_tokenizer = PoseTokenizer(vqvae)
    w2v = api.load("word2vec-google-news-300")
    text_tokenizer = Word2VecTokenizer.from_word2vec(w2v) #type:ignore

    # 3. Crei il layer per il tuo Transformer
    factory = Word2VecWeightFactory(text_tokenizer, w2v)
    text_embedding = factory.get_nn_embedding(freeze=True)
    # ===== 2. Crea Dataset e DataLoader =====
    print("Loading datasets...")
    train_dataset = TranscriptPoseDataset(
        speakers=config['speakers'],
        data_root=config['data_root'],
        pose_tokenizer=pose_tokenizer,
        text_tokenizer=text_tokenizer,
        split="train",
        cache_path=config['caches_path'],
        max_len=hyperparams['max_seq_len']
    )
    
    val_dataset = TranscriptPoseDataset(
        speakers=config['speakers'],
        data_root=config['data_root'],
        pose_tokenizer=pose_tokenizer,
        text_tokenizer=text_tokenizer,
        split="dev",
        cache_path=config['caches_path'],
        max_len=hyperparams['max_seq_len']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        collate_fn=transcript_motion_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=False,
        collate_fn=transcript_motion_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    hyperparams = { **hyperparams,
                   'pose_vocab_size': pose_tokenizer.vocab_size,
                   'text_embedding': text_embedding,
                   }
    
    # ===== 3. Crea Modello =====
    print("Creating GestureTransformer model...")
    model = XGestureTransformer(
        **hyperparams
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== 4. Optimizer, Scheduler, Loss =====
    optimizer = AdamW(model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
    # scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    # CrossEntropy con ignore_index per ignorare i token di padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD_TOKEN = 0
    
    # ===== 5. Training Loop =====
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 15
    try:
        for epoch in range(1, config['epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{config['epochs']}")
            print(f"{'='*50}")
            
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Salva checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model.save(output_dir / "best_model.pt")
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1
            
            # Salva checkpoint periodico
            if epoch % config['save_every'] == 0:
                print(f"Saving checkpoint at epoch {epoch}...")
                model.save(output_dir / f"checkpoint.pt")
            
            if epochs_no_improve >= patience:
                print(f"Train stopped at epoch {epoch}")
                break

            run.log({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Learning Rate": scheduler.get_last_lr()[0],
            })
        
        # Salva modello finale
        # model.save(output_dir / "final_model.pt")
        print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}")
    finally:
        run.finish()


if __name__ == "__main__":
    speakers = [ "almaram", "chemistry", "corden", "huckabee", "lec_evol", "maher", "oliver", "shelly", "ytch_prof",
        "angelica",  "colbert", "ellen", "jon", "lec_hist", "minhaj", "rock", "ytch_charisma",
        "bee", "conan", "fallon", "lec_cosmic", "lec_law", "noah", "seth", "ytch_dating"
        ]
    config = {
        'speakers': speakers,
        'data_root': PRJ_ROOT / 'dataset/pats/data',          # Percorso ai dati
        'pose_tokenizer_path': PRJ_ROOT / 'weights/vqvae/vqvae_trial_8.pt',
        'text_tokenizer_path': PRJ_ROOT / 'caches/tokenizers/word2vec_tokenizer.json',
        'caches_path': PRJ_ROOT / 'caches/dataset/',
        'output_dir': PRJ_ROOT / 'weights/transformer/x_drop/',
        'num_workers': 4,
        'epochs': 100,
        'save_every': 10,
    }

    wandb.login(key=WANDB_API_KEY)

    main(config)
