import numpy as np
from sklearn.metrics import accuracy_score,hamming_loss,precision_score,recall_score,f1_score,classification_report
from torch.utils.data import SubsetRandomSampler, DataLoader
import re
import nltk
import string


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



# stopwords + punctuation
stop_words = set(stopwords.words('english')).union(set(string.punctuation)) 




################## preprocessing text #########################
def preprocess(text):

  words = word_tokenize(text)
  filtered_sentence = [] 
  # remove stopwords
  for word in words: 
    if word not in stop_words: 
        filtered_sentence.append(word) 
  text = ' '.join(filtered_sentence)
  # lemmatize
  # lemma_word = []
  # wordnet_lemmatizer = WordNetLemmatizer()
  # for w in filtered_sentence:
  #   word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
  #   word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
  #   word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
  #   lemma_word.append(word3)
  return text

#######################################################################


###################### Calculation of Metrics #########################
def calculate_metrics(pred, target, threshold=0.5):
  pred = np.array(pred > threshold, dtype="float32")
  
  return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'hammingloss':hamming_loss(target,pred)       
        }
#########################################################################

################### Label One-hot Encodings ########################
def labeltarget(x,frequent_list):
  target=np.zeros(10,dtype="float32")
  for index,code in enumerate(frequent_list):
    if code in x :
      target[index]=1
  return target
#####################################################################

#######################################################################################
def split_indices(dataset, validation_split, shuffle_dataset = True, random_seed = 2021):
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  return indices[split:], indices[:split]
#########################################################################################

#########################################################################################
def dataloader(train_dataset, test_dataset, batch_size, val_split):
  train_indices, val_indices = split_indices(train_dataset, val_split)
  train_sampler = SubsetRandomSampler(train_indices)
  val_sampler = SubsetRandomSampler(val_indices)
  train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler=train_sampler)
  val_loader = DataLoader(train_dataset, batch_size = batch_size, sampler=val_sampler)
  test_loader = DataLoader(test_dataset, batch_size= batch_size)
  return train_loader, val_loader, test_loader


#########################################################################################

#########################################################################################
def train_metric(y_pred, y_test, threshold=0.5):
  num_classes = y_pred.shape[1]
  y_pred_tags = (y_pred>0.5).float()

  correct_pred = (y_pred_tags == y_test).float()
  accuracy = (correct_pred.sum(dim=1) == num_classes).float().sum() / len(correct_pred)

  hammingloss = hamming_loss(y_test.cpu().numpy(), y_pred_tags.cpu().numpy())

  f1score = f1_score(y_true=y_test.cpu().numpy(), y_pred=y_pred_tags.cpu().numpy(), average='micro')
  return accuracy, hammingloss, f1score

#################################################################################################