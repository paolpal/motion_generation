from motion_generation.tokenizer import Word2VecTokenizer

if __name__ == "__main__":
    import gensim.downloader as api
    w2v = api.load("word2vec-google-news-300")
    tokenizer = Word2VecTokenizer.from_word2vec(w2v)
    print(tokenizer.summary())    
    tokenizer.save("word_tokenizer.json")
    loaded_tokenizer = Word2VecTokenizer.load("word_tokenizer.json")
    print(loaded_tokenizer.summary())