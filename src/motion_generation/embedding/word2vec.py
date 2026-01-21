import torch
import torch.nn as nn
import numpy as np

from motion_generation.tokenizer import Word2VecTokenizer

class Word2VecWeightFactory:
    """
    Gestisce la creazione della matrice dei pesi coerente con il Word2VecTokenizer.
    """
    def __init__(self, tokenizer: Word2VecTokenizer, word2vec_model):
        self.tokenizer = tokenizer
        self.wv = word2vec_model
        self.vector_size = word2vec_model.vector_size

    def build_matrix(self):
        """
        Costruisce la matrice dei pesi seguendo l'ordine stoi del tokenizer.
        """
        vocab_size = self.tokenizer.vocab_size
        # Inizializziamo con zeri (importante per il PAD all'indice 0)
        matrix = torch.zeros(vocab_size, self.vector_size)
        
        # 1. SIL (Index 1): Inizializzazione casuale (verrà appresa)
        # Usiamo una deviazione standard piccola per non "sparare" gradienti troppo forti all'inizio
        matrix[self.tokenizer.sil_id] = torch.randn(self.vector_size) * 0.02
        
        # 2. OOV (Index 2): Media dei vettori Word2Vec
        # Usiamo la media dei vettori esistenti per dare una base semantica neutra
        avg_vector = np.mean(self.wv.vectors, axis=0)
        matrix[self.tokenizer.oov_id] = torch.from_numpy(avg_vector)
        
        # 3. Parole (Index 3+): Copiamo i pesi da Word2Vec
        found = 0
        for word, idx in self.tokenizer.stoi.items():
            # Saltiamo i token speciali che abbiamo già gestito
            if word in self.tokenizer.SPECIAL_TOKENS:
                continue
                
            if word in self.wv:
                # Usiamo .copy() per evitare problemi di sola lettura di Gensim
                matrix[idx] = torch.from_numpy(self.wv[word].copy())
                found += 1
                
        print(f"Matrice pesi costruita. Parole trovate in Word2Vec: {found}/{vocab_size - 3}")
        return matrix

    def get_nn_embedding(self, freeze=False):
        """
        Crea e ritorna un modulo nn.Embedding pronto all'uso.
        """
        weights = self.build_matrix()
        embedding = nn.Embedding.from_pretrained(
            weights,
            freeze=freeze,
            padding_idx=self.tokenizer.pad_id
        )
        return embedding

# --- ESEMPIO DI UTILIZZO ---
# 1. Carichi il tokenizer salvato
# tokenizer = Word2VecTokenizer.load("tokenizer.json")

# 2. Carichi Word2Vec (o ne hai uno già in memoria)
# wv = api.load('word2vec-google-news-300')

# 3. Crei il layer per il tuo Transformer
# factory = Word2VecWeightFactory(tokenizer, wv)
# self.text_embedding = factory.get_nn_embedding(freeze=False)