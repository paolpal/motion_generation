# 1. Carichi il tokenizer salvato
import gensim.downloader as api
from motion_generation.embedding import Word2VecWeightFactory
from motion_generation.tokenizer import Word2VecTokenizer


PRJ_ROOT = "/home/paolo/Projects/Posemi/motion_generation"
tokenizer = Word2VecTokenizer.load(f"{PRJ_ROOT}/weights/tokenizer/word_tokenizer.json")

# 2. Carichi Word2Vec (o ne hai uno già in memoria)
wv = api.load('word2vec-google-news-300')

# 3. Crei il layer per il tuo Transformer
factory = Word2VecWeightFactory(tokenizer, wv)
text_embedding = factory.get_nn_embedding(freeze=False)

print(text_embedding.shape)