import os
import random

import wandb

from motion_generation.embedding import Word2VecWeightFactory
os.environ["GENSIM_DATA_DIR"] = "/home/ubuntu/palumbo/Posemi/dataset/gensim_data"
import gensim.downloader as api
import torch
from pathlib import Path
from tqdm import tqdm

from motion_generation.models import XGestureDecoder
from motion_generation.dataset import TranscriptPoseDataset


from motion_quantization.models import SkeletonVQVAE
from motion_quantization.quantization import PoseTokenizer
from motion_generation.tokenizer import Word2VecTokenizer

from motion_generation.scripts.render_video import render_video

from torch.nn.utils.rnn import pad_sequence
from motion_generation.scripts.save_attn import save_attention_heatmap, save_attention_subplots 

PRJ_ROOT = Path("/home/ubuntu/palumbo/Posemi/")

@torch.no_grad()
def main(config):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading tokenizers and dataset...")
    vqvae = SkeletonVQVAE.load(config['pose_tokenizer_path'])
    pose_tokenizer = PoseTokenizer(vqvae)
    w2v = api.load("word2vec-google-news-300")
    text_tokenizer = Word2VecTokenizer.from_word2vec(w2v) #type:ignore

    factory = Word2VecWeightFactory(text_tokenizer, w2v)
    text_embedding = factory.get_nn_embedding(freeze=True)
    
    print("Loading model...")
    model = XGestureDecoder.load( config['gesture_transformer_path'] / "best_model.pt", text_embedding=text_embedding, device=device)
    model.eval()
    print("Model loaded.")

    val_dataset = TranscriptPoseDataset(
        speakers=config['speakers'],
        data_root=config['data_root'],
        pose_tokenizer=pose_tokenizer,
        text_tokenizer=text_tokenizer,
        split="dev",
        cache_path=config['caches_path']
    )

    print("Tokenizers and dataset loaded.")

    rand_idx = random.randint(0, len(val_dataset) - 1)
    # rand_idx = 5266
    print(f"Generating gesture for sample index: {rand_idx}")

    text_tokens, motion_tokens = val_dataset[rand_idx]
    
    gest_seq = torch.tensor([pose_tokenizer.som_id], dtype=torch.long)

    cross_attn_maps = []

    bar = tqdm(range(1, motion_tokens.shape[0]), desc="Generating gesture sequence") 
    for i in bar:
        encoder_input = text_tokens.unsqueeze(0).to(device)           # (1, T)
        decoder_input = gest_seq.unsqueeze(0).to(device)              # (1, M)

        # bar.set_postfix({'Shape': encoder_input[:,:i+look_ahead].shape})

        logits, attn = model(
            src=encoder_input[:,:i],
            tgt=decoder_input,
            src_padding_mask=None,
            tgt_padding_mask=None,
            return_attn=True
        )  # (1, M, vocab_size)

        cross_attn_maps.append(attn[-1].detach().cpu())

        probs = torch.softmax(logits[0, -1, :], dim=-1)

        next_token_id = torch.multinomial(probs, num_samples=1)  # (1,)
        
        #next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0)  # (1,)

        gest_seq = torch.cat([gest_seq, next_token_id.cpu()], dim=0)

        if next_token_id.item() == pose_tokenizer.eom_id:
            bar.set_postfix({'Status': 'EOM reached'})
            break

    # print("Generated gesture token IDs:", gest_seq.tolist())
    # print("Target gesture token IDs:   ", motion_tokens.tolist())

    text = [text_tokenizer.decode(word_id) for word_id in text_tokens.tolist() if word_id != text_tokenizer.pad_id]

    text = [""] + text

    # =========================
    # VISUALIZZAZIONE DELLE ATTENTION
    # =========================

    attn = cross_attn_maps[-1]  # (1, num_heads, tgt_len, src_len)
    # print(len(attn))
    # print(attn.shape)
    attn = attn.squeeze(0)
    attn_mean = attn.mean(dim=0) # (tgt_len, src_len)
    # mask = model.get_gestural_mask(attn_mean.size(0), attn_mean.size(1), device=attn_mean.device).squeeze(0).squeeze(0)
    all_true = torch.ones_like(attn_mean, dtype=torch.bool)

    mask = all_true

    row_max = torch.where(mask, attn_mean, torch.tensor(float('-inf'))).max(dim=1, keepdim=True).values
    row_min = torch.where(mask, attn_mean, torch.tensor(float('inf'))).min(dim=1, keepdim=True).values
    denom = row_max - row_min + 1e-8
    attn_vis = (attn_mean - row_min) / denom
    attn_vis = torch.where(mask, attn_vis, torch.tensor(float('NaN')))

    # 3. Salvataggio della heatmap finale
    save_attention_heatmap(
        attn_vis, 
        x_labels=[text_tokenizer.decode(word_id) for word_id in text_tokens.tolist() if word_id != text_tokenizer.pad_id],
        out_path=PRJ_ROOT / "images/mean_attention.png"
    )

    print(len(cross_attn_maps))

    attn = cross_attn_maps[-1]
    # print(attn.shape)
    attn = attn.squeeze(0)
    attns = []
    for i in range(attn.size(0)):
        attn_head = attn[i]  # (tgt_len, src_len)
        # mask = model.get_gestural_mask(attn_mean.size(0), attn_mean.size(1), device=attn_mean.device).squeeze(0).squeeze(0)

        all_true = torch.ones_like(attn_head, dtype=torch.bool)
        mask = all_true

        row_max = torch.where(mask, attn_head, torch.tensor(float('-inf'))).max(dim=1, keepdim=True).values
        row_min = torch.where(mask, attn_head, torch.tensor(float('inf'))).min(dim=1, keepdim=True).values
        denom = row_max - row_min + 1e-8
        attn_norm = (attn_head - row_min) / denom
        attn_norm = torch.where(mask, attn_norm, torch.tensor(float('NaN')))

        attns.append(attn_norm)

    save_attention_subplots(
        attns,
        out_path=PRJ_ROOT / "images/all_heads_attention.png"
    )
        
    # =========================
    # VISUALIZZAZIONE DELLE ATTENTION
    # =========================

    render_video(motion_tokens, gest_seq, pose_tokenizer, text=text, output_filename=PRJ_ROOT / "videos/gesture_comparison.mp4")

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
        'gesture_transformer_path': PRJ_ROOT / 'weights/transformer/x_decoder/',
        'caches_path': PRJ_ROOT / 'caches/dataset/',
    }

    main(config)