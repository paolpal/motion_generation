import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import argparse

from motion_generation.models.gesture import GestureTransformer
from motion_generation.dataset.transcript_pose import TranscriptPoseDataset, transcript_motion_collate_fn
from motion_generation.embedding import Word2VecEmbedder
from motion_quantization.quantization import PoseQuantizer
from motion_quantization.models import SkeletonVQVAE


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        encoder_input = batch['encoder_input'].to(device)           # (B, T, 300)
        decoder_input = batch['decoder_input'].to(device)           # (B, M)
        decoder_target = batch['decoder_target'].to(device)         # (B, M)
        encoder_padding_mask = batch['encoder_padding_mask'].to(device)
        decoder_padding_mask = batch['decoder_padding_mask'].to(device)
        
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
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        encoder_padding_mask = batch['encoder_padding_mask'].to(device)
        decoder_padding_mask = batch['decoder_padding_mask'].to(device)
        
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


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== 1. Carica Quantizer e Tokenizer =====
    print("Loading PoseQuantizer...")
    model = SkeletonVQVAE.load("/home/paolo/Projects/Posemi/motion_quantization/scripts/weights/vqvae_trial_8.pt")
    model.to(device)
    pose_quantizer = PoseQuantizer(model=model, device=device)
    
    print("Loading Word2Vec Embedder...")
    tokenizer = Word2VecEmbedder()
    
    # Calcola vocab_size: codebook + 3 token speciali (PAD, BOS, EOS)
    vocab_size = pose_quantizer.codebook.size(0) + 3
    print(f"Vocab size: {vocab_size} (codebook: {pose_quantizer.codebook.size(0)} + 3 special tokens)")
    
    # ===== 2. Crea Dataset e DataLoader =====
    print("Loading datasets...")
    train_dataset = TranscriptPoseDataset(
        speakers=args.speakers,
        data_root=args.data_root,
        pose_quantizer=pose_quantizer,
        tokenizer=tokenizer,
        split="train"
    )
    
    val_dataset = TranscriptPoseDataset(
        speakers=args.speakers,
        data_root=args.data_root,
        pose_quantizer=pose_quantizer,
        tokenizer=tokenizer,
        split="dev"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=transcript_motion_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=transcript_motion_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ===== 3. Crea Modello =====
    print("Creating GestureTransformer model...")
    model = GestureTransformer(
        vocab_size=vocab_size,
        d_text=300,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== 4. Optimizer, Scheduler, Loss =====
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # CrossEntropy con ignore_index per ignorare i token di padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD_TOKEN = 0
    
    # ===== 5. Training Loop =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Salva checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output_dir / "best_model.pt")
            print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
        
        # Salva checkpoint periodico
        if epoch % args.save_every == 0:
            model.save(output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Salva modello finale
    model.save(output_dir / "final_model.pt")
    print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = {
        'speakers': ['speaker1', 'speaker2'],  # Esempio di speaker
        'data_root': '/path/to/data',          # Percorso ai dati
        'batch_size': 16,
        'num_workers': 4,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'output_dir': './gesture_transformer_output',
        'save_every': 10
    }
    main(args)
