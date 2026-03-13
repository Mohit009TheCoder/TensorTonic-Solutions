import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    
    # Tokenize documents
    tokenized_docs = [doc.split() for doc in documents]
    
    # Build vocabulary
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word:i for i,word in enumerate(vocab)}
    
    N = len(documents)
    V = len(vocab)
    
    # Compute document frequency
    df = {word:0 for word in vocab}
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1
    
    # Compute IDF
    idf = {word: math.log(N / df[word]) for word in vocab}
    
    # Build TF-IDF matrix
    tfidf_matrix = np.zeros((N, V))
    
    for i, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        total_terms = len(doc)
        
        for word, count in counts.items():
            j = vocab_index[word]
            tf = count / total_terms
            tfidf_matrix[i, j] = tf * idf[word]
    
    return tfidf_matrix, vocab


docs = ["the cat sat","the cat ran","the dog sat"]

matrix, vocab = tfidf_vectorizer(docs)

print("Vocabulary:", vocab)
print("TF-IDF Matrix:\n", matrix)

docs1 = ["machine learning is great","cooking pasta is fun"]
matrix, vocab = tfidf_vectorizer(docs1)

print("Vocabulary:", vocab)
print("TF-IDF Matrix:\n", matrix)
