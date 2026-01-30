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
from motion_generation.dataset import TranscriptPoseDataset, transcript_motion_collate_fn


from motion_quantization.models import SkeletonVQVAE
from motion_quantization.quantization import PoseTokenizer
from motion_generation.tokenizer import Word2VecTokenizer

from motion_generation.models import GestureTransformer

from motion_generation.scripts.render_video import render_video

from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
from motion_generation.scripts.save_attn import save_attention_heatmap 

PRJ_ROOT = Path("/home/ubuntu/palumbo/Posemi/")

config = {
    'gesture_transformer_path': PRJ_ROOT / "weights/transformer/test_save_reload",
    'pose_tokenizer_path': PRJ_ROOT / 'weights/vqvae/vqvae_trial_8.pt',
}

hyperparams = {
    'nhead': 4,
    'encoder_layers': 4,
    'decoder_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'look_ahead_steps': 3,
    'max_seq_len': 500,
    'memory_causal': True,
}

# ===== 1. Carica Pose e Text Tokenizer e Text Embedding =====

vqvae = SkeletonVQVAE.load(config['pose_tokenizer_path'])
pose_tokenizer = PoseTokenizer(vqvae)
w2v = api.load("word2vec-google-news-300")
text_tokenizer = Word2VecTokenizer.from_word2vec(w2v) #type:ignore

# 3. Crei il layer per il tuo Transformer
factory = Word2VecWeightFactory(text_tokenizer, w2v)
text_embedding = factory.get_nn_embedding(freeze=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = GestureTransformer(
        text_embedding=text_embedding,
        transcript_vocab_size=text_tokenizer.vocab_size,
        pose_vocab_size=pose_tokenizer.vocab_size,
        d_model=text_embedding.embedding_dim,
        nhead=hyperparams['nhead'],
        num_encoder_layers=hyperparams['encoder_layers'],
        num_decoder_layers=hyperparams['decoder_layers'],
        dim_feedforward=hyperparams['dim_feedforward'],
        dropout=hyperparams['dropout'],
        look_ahead_steps=hyperparams['look_ahead_steps'],
        max_seq_len=hyperparams['max_seq_len'],
        memory_causal=hyperparams['memory_causal']
    ).to(device)

model.save(config['gesture_transformer_path'] / "model.pt")

# ===== 2. Ricarica il modello =====
text_embedding = factory.get_nn_embedding(freeze=True)

reloaded_model = GestureTransformer.load(config['gesture_transformer_path'] / "model.pt", text_embedding=text_embedding, device=device)    

