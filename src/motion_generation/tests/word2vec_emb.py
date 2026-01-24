# 1. Carichi il tokenizer salvato
import os
os.environ["GENSIM_DATA_DIR"] = "/home/ubuntu/palumbo/Posemi/dataset/gensim_data"
import gensim.downloader as api
from motion_generation.embedding import Word2VecWeightFactory
from motion_generation.tokenizer import Word2VecTokenizer


PRJ_ROOT = "/home/ubuntu/palumbo/Posemi/"

w2v = api.load("word2vec-google-news-300")
tokenizer = Word2VecTokenizer.from_word2vec(w2v) #type:ignore

# 3. Crei il layer per il tuo Transformer
factory = Word2VecWeightFactory(tokenizer, w2v)
text_embedding = factory.get_nn_embedding()

print(text_embedding.weight.shape)

print(text_embedding.weight[tokenizer.pad_id].shape) # PAD
print(text_embedding.weight[tokenizer.sil_id].shape) # SIL
print(text_embedding.weight[tokenizer.oov_id].shape) # OOV

print(tokenizer.pad_id, tokenizer.sil_id, tokenizer.oov_id)

text_embedding.weight.requires_grad = False  
text_embedding.weight[tokenizer.sil_id].requires_grad = True
text_embedding.weight[tokenizer.oov_id].requires_grad = True

print(text_embedding.weight.requires_grad)
print(text_embedding.weight[:3].requires_grad)
print(text_embedding.weight[3:6].requires_grad)

