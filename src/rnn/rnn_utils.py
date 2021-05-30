import numpy as np 
from collections import Counter
import pandas as pd 
import re
import nltk
import string
import torch
import torch.nn as nn
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')).union(set(string.punctuation)) 

def preprocessing_rnn(text):
  words=word_tokenize(text)
  filtered_sentence = [] 
  # remove stopwords
  for word in words: 
    if word not in stop_words: 
        filtered_sentence.append(word) 
  
  # lemmatize
  lemma_word = []
  wordnet_lemmatizer = WordNetLemmatizer()
  for w in filtered_sentence:
    word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
    word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
    word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
    lemma_word.append(word3)
  return lemma_word


def count_vocab_index(train_df, test_df):
    df = pd.concat([train_df, test_df]).sample(frac=1).reset_index(drop=True)
    counts = Counter()
    for _, row in df.iterrows():
        counts.update(preprocessing_rnn(row['discharge_diagnosis']))
    
    # removing the words that have frequency less than 2
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    
    vocab2index = {"":0, "UNKNOWN":1}
    words = ["", "UNKNOWN"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    return counts, vocab2index


def encode_sentence(text, vocab2index, N = 50):
  tokenized = preprocessing_rnn(text)
  encoded = np.zeros(N, dtype=int)
  enc1 = np.array([vocab2index.get(word, vocab2index["UNKNOWN"]) for word in tokenized])
  length = min(N, len(enc1))
  encoded[:length] = enc1[:length]
  return encoded
        


def get_emb_matrix(w2vmodel, word_counts):
  """ Creates embedding matrix from word vectors"""
  vocab_size = len(word_counts) + 2
  emb_size = w2vmodel.vector_size

  W = np.zeros((vocab_size, emb_size), dtype="float32")
  W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
  W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
  
  i = 2
  for word in word_counts:
    if word in w2vmodel.wv:
      W[i] = w2vmodel.wv[word]
    else:
      W[i] = np.random.uniform(-0.25,0.25, emb_size)
    i += 1   
  return W

def create_emb_layer(weights_matrix, non_trainable=False):
  num_embeddings, embedding_dim = weights_matrix.shape
  emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
  emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
  if non_trainable:
    emb_layer.weight.requires_grad = False

  return emb_layer, num_embeddings, embedding_dim
