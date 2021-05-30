import numpy as np


def character_index(sentence, vocabulary, sequence_length = 500):
  index_list = []
  for i in range(len(sentence)):
    if i > sequence_length-1:
      break
    else:
      if sentence[i] in vocabulary:
        index_list.append(vocabulary.index(sentence[i]) + 1)
      else :
        index_list.append(len(vocabulary)+1)
  if len(index_list) == sequence_length:
    return index_list
  else:
    index_list.extend([0]*(sequence_length-len(index_list)))
    return index_list

def character_embedding(index_list, vocabulary):
  embedding_weights = []
  for index,i in enumerate(index_list):
    one_hot = np.zeros(len(vocabulary)+1)
    if i != 0:
      one_hot[i-1] = 1
    embedding_weights.append(one_hot)
  return np.array(embedding_weights,dtype = 'float32').T