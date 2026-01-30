import os
import random

import wandb

from motion_generation.embedding import Word2VecWeightFactory
os.environ["GENSIM_DATA_DIR"] = "/home/ubuntu/palumbo/Posemi/dataset/gensim_data"
import gensim.downloader as api
import torch
from pathlib import Path
from tqdm import tqdm

from motion_generation.models import GestureTransformer
from motion_generation.dataset import TranscriptPoseDataset


from motion_quantization.models import SkeletonVQVAE
from motion_quantization.quantization import PoseTokenizer
from motion_generation.tokenizer import Word2VecTokenizer

from motion_generation.models import GestureTransformer

from motion_generation.scripts.render_video import render_video

from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
from motion_generation.scripts.save_attn import save_attention_heatmap, save_attention_subplots 


# =========================
# FORZA PyTorch A RESTITUIRE LE ATTENTION
# =========================

_original_mha_forward = MultiheadAttention.forward

def patched_mha_forward(self, query, key, value, **kwargs):
    kwargs["need_weights"] = True
    kwargs["average_attn_weights"] = False
    return _original_mha_forward(self, query, key, value, **kwargs)

MultiheadAttention.forward = patched_mha_forward # type: ignore 

# =========================
# FORZA PyTorch A RESTITUIRE LE ATTENTION
# =========================

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
    model = GestureTransformer.load( config['gesture_transformer_path'] / "best_model.pt", text_embedding=text_embedding, device=device)
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
    rand_idx = 5266
    print(f"Generating gesture for sample index: {rand_idx}")

    text_tokens, motion_tokens = val_dataset[rand_idx]
    
    gest_seq = torch.tensor([pose_tokenizer.som_id], dtype=torch.long)

    # =========================
    # FORZA PyTorch A RESTITUIRE LE ATTENTION
    # =========================

    cross_attn_maps = []

    def cross_attn_hook(module, input, output):
        # output = (attn_output, attn_weights)
        attn_weights = output[1]
        if attn_weights is not None:
            cross_attn_maps.append(attn_weights.detach().cpu())

    handle = model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook( # type: ignore 
        cross_attn_hook
    )

    # =========================
    # FORZA PyTorch A RESTITUIRE LE ATTENTION
    # =========================

    bar = tqdm(range(1, motion_tokens.shape[0]), desc="Generating gesture sequence") 
    for i in bar:
        encoder_input = text_tokens.unsqueeze(0).to(device)           # (1, T)
        decoder_input = gest_seq.unsqueeze(0).to(device)              # (1, M)

        bar.set_postfix({'Shape': encoder_input[:,:i+3].shape})

        logits = model(
            src=encoder_input[:,:i+3],
            tgt=decoder_input,
            src_padding_mask=None,
            tgt_padding_mask=None
        )  # (1, M, vocab_size)

        probs = torch.softmax(logits[0, -1, :], dim=-1)

        next_token_id = torch.multinomial(probs, num_samples=1)  # (1,)
        
        #next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0)  # (1,)

        gest_seq = torch.cat([gest_seq, next_token_id.cpu()], dim=0)

        if next_token_id.item() == pose_tokenizer.eom_id:
            break

    # print("Generated gesture token IDs:", gest_seq.tolist())
    # print("Target gesture token IDs:   ", motion_tokens.tolist())

    text = [text_tokenizer.decode(word_id) for word_id in text_tokens.tolist() if word_id != text_tokenizer.pad_id]

    text = [""] + text

    # =========================
    # FORZA PyTorch A RESTITUIRE LE ATTENTION
    # =========================
    handle.remove()  # importantissimo

    attn = cross_attn_maps[-1][0]  # (num_heads, tgt_len, src_len)
    attn_mean = attn.mean(dim=0) # (tgt_len, src_len)
    mask = ~model.get_look_ahead_causal_mask(attn_mean.size(0))

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

    attn = cross_attn_maps[-1][0]
    print(attn.shape)
    attns = []
    for i in range(attn.size(0)):
        attn_head = attn[i]  # (tgt_len, src_len)
        model.context_back_steps = 18
        mask = ~model.get_sliding_window_mask(attn_head.size(0))

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
    # FORZA PyTorch A RESTITUIRE LE ATTENTION
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
        'gesture_transformer_path': PRJ_ROOT / 'weights/transformer/reduced/',
        'caches_path': PRJ_ROOT / 'caches/dataset/',
    }

    main(config)