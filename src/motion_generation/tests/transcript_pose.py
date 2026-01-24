import torch
from torch.utils.data import Dataset, DataLoader
from motion_generation.dataset import TranscriptPoseDataset, transcript_motion_collate_fn
from motion_quantization.quantization import PoseTokenizer
from motion_quantization.models import SkeletonVQVAE
from motion_generation.tokenizer import Word2VecTokenizer

from gensim.models.keyedvectors import KeyedVectors

import numpy as np

import os
os.environ["GENSIM_DATA_DIR"] = "/home/ubuntu/palumbo/Posemi/dataset/gensim_data"
import gensim.downloader as api

def decode_text_tokens(tokens, tokenizer: Word2VecTokenizer):
    words = []
    for token in tokens:
        word = tokenizer.itos[token.item()]
        words.append(word)
    return ' '.join(words)

ROOT_PRJ = "/home/ubuntu/palumbo/Posemi/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Pose Tokenizer...")
vqvae = SkeletonVQVAE.load(f"{ROOT_PRJ}/weights/vqvae/vqvae_trial_8.pt")
pose_tokenizer = PoseTokenizer(model=vqvae, device=device)

print("Loading Text Tokenizer...")
tokenizer_cache_path = f"{ROOT_PRJ}/caches/tokenizers/word2vec_tokenizer.json"
if os.path.exists(tokenizer_cache_path):
    text_tokenizer = Word2VecTokenizer.load(tokenizer_cache_path)
else:
    w2v : KeyedVectors = api.load("word2vec-google-news-300") #type:ignore
    text_tokenizer = Word2VecTokenizer.from_word2vec(w2v)
    text_tokenizer.save(tokenizer_cache_path)

speakers = ["almaram", "chemistry", "corden", "huckabee", "lec_evol", "maher", "oliver", "shelly", "ytch_prof",
"angelica",  "colbert", "ellen", "jon", "lec_hist", "minhaj", "rock", "ytch_charisma",
"bee", "conan", "fallon", "lec_cosmic", "lec_law", "noah", "seth", "ytch_dating"]

split = "train"

print("Creating TranscriptPoseDataset...")
trainset = TranscriptPoseDataset(
    speakers=speakers,
    split=split,
    data_root=f"{ROOT_PRJ}/dataset/pats/data",
    pose_tokenizer=pose_tokenizer,
    text_tokenizer=text_tokenizer,
    cache_path=f"{ROOT_PRJ}/caches/dataset/",
    force_rebuild=False
)
print("--- Dataset created ---")

print("Dataset length:", len(trainset))

idx = 42

text_tokens, motion_tokens = trainset[idx]
print("Text: ", decode_text_tokens(text_tokens, text_tokenizer))
print("Text Tokens shape:", text_tokens.shape)
# print("Words:", clip['words'])
# print("Motion Tokens:", motion_tokens)
print("Motion Tokens shape:", motion_tokens.shape)


sil_found = 0
oov_found = 0
total_text_len = 0
max_text_len = 0
min_text_len = len(trainset[0][0])
for i in range(len(trainset)):
    text_tokens, motion_tokens = trainset[i]

    if text_tokenizer.sil_id in text_tokens:
        sil_found += 1

    if text_tokenizer.oov_id in text_tokens:
        oov_found += 1
    total_text_len += len(text_tokens)
    max_text_len = max(max_text_len, len(text_tokens))
    min_text_len = min(min_text_len, len(text_tokens))

print(f"Samples with SIL token: {sil_found} / {len(trainset)}")
print(f"Samples with OOV token: {oov_found} / {len(trainset)}")
print("")
print(f"Average text length: {total_text_len / len(trainset):.2f} tokens")
print(f"Max text length: {max_text_len} tokens")
print(f"Min text length: {min_text_len} tokens")


width = max_text_len - min_text_len
twentyfive_percentile = int(0.25 * width) + min_text_len
fifty_percentile = int(0.50 * width) + min_text_len
seventyfive_percentile = int(0.75 * width) + min_text_len

count_25 = 0
count_50 = 0
count_75 = 0
lens = []
for i in range(len(trainset)): 
    text_tokens, motion_tokens = trainset[i]
    l = len(text_tokens)
    if l >= twentyfive_percentile:
        count_25 += 1
    if l >= fifty_percentile:
        count_50 += 1
    if l >= seventyfive_percentile:
        count_75 += 1

    lens.append(l)

p25 = np.percentile(lens, 25)
p50 = np.percentile(lens, 50)
p75 = np.percentile(lens, 75)
p80 = np.percentile(lens, 80)
p85 = np.percentile(lens, 85)
p90 = np.percentile(lens, 90)
p95 = np.percentile(lens, 95)
p99 = np.percentile(lens, 99)

print(f"25th Percentile: {int(p25)} tokens")
print(f"50th Percentile: {int(p50)} tokens")
print(f"75th Percentile: {int(p75)} tokens")
print(f"80th Percentile: {int(p80)} tokens")
print(f"85th Percentile: {int(p85)} tokens")
print(f"90th Percentile: {int(p90)} tokens")
print(f"95th Percentile: {int(p95)} tokens")
print(f"99th Percentile: {int(p99)} tokens")


trainloader = DataLoader(trainset, batch_size=8, collate_fn=transcript_motion_collate_fn)
for batch in trainloader:
    text_tokens_batch = batch['text_tokens']
    motion_tokens_batch = batch['motion_tokens']
    
    print("Batch Text Tokens shape:", text_tokens_batch.shape)
    print("Batch Motion Tokens shape:", motion_tokens_batch.shape)
    break