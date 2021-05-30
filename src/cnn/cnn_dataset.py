import torch 
from torch.utils.data import Dataset
from src.cnn.cnn_utils import character_embedding, character_index
from src.utils import labeltarget

frequent_icd9category = ['401','427','276','414','272','250','428','518','285','584']
frequent_icd9code = ['4019', '4280', '42731', '41401', '5849', '25000', '2724', '51881', '5990', '53081']
frequent_icd10category = ['I10', 'I25', 'E78', 'I50', 'I48', 'N17', 'E87', 'E11', 'J96', 'N39']
frequent_icd10code = ['I10', 'I2510', 'I509', 'I4891', 'N179', 'E119', 'E784', 'E785', 'J9690', 'J9600']

class cnndataset(Dataset):
  def __init__(self, df, vocabulary = """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'"/\|_@#$%^&*~`+-=<>()[]{}""", sequence_length = 140):
    self.df = df
    self.nsamples = len(df)
    self.vocabulary = list(vocabulary)
    self.sequence_length = sequence_length
    

  def __getitem__(self,index):
        
    x = torch.from_numpy(character_embedding(character_index(self.df['discharge_diagnosis'].iloc[index], self.vocabulary, self.sequence_length), self.vocabulary))
    y = {}
    y['icd9code'] = torch.from_numpy(labeltarget(self.df["ICD9_CODE"].iloc[index], frequent_icd9code))
    y['icd9cat'] = torch.from_numpy(labeltarget(self.df["ICD9_CATEGORY"].iloc[index], frequent_icd9category))
    y['icd10code'] = torch.from_numpy(labeltarget(self.df["ICD10"].iloc[index], frequent_icd10code))
    y['icd10cat'] = torch.from_numpy(labeltarget(self.df["ICD10_CATEGORY"].iloc[index], frequent_icd10category))
    return x, y

  def __len__(self):
    return self.nsamples