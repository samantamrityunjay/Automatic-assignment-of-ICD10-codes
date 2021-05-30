import torch 
from torch.utils.data import Dataset
import transformers

from src.utils import preprocess, labeltarget

frequent_icd9category = ['401','427','276','414','272','250','428','518','285','584']
frequent_icd9code = ['4019', '4280', '42731', '41401', '5849', '25000', '2724', '51881', '5990', '53081']
frequent_icd10category = ['I10', 'I25', 'E78', 'I50', 'I48', 'N17', 'E87', 'E11', 'J96', 'N39']
frequent_icd10code = ['I10', 'I2510', 'I509', 'I4891', 'N179', 'E119', 'E784', 'E785', 'J9690', 'J9600']

class BERTdataset(Dataset):
    def __init__(self, df, max_len=128):
        self.df = df
        self.nsamples = len(df)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = max_len

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        text = self.df["discharge_diagnosis"].iloc[idx]
        inputs = self.tokenizer.encode_plus(
            text=preprocess(text),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        target = {
            "icd9code": torch.from_numpy(
                labeltarget(self.df["ICD9_CODE"].iloc[idx], frequent_icd9code)
            ),
            "icd9cat": torch.from_numpy(
                labeltarget(self.df["ICD9_CATEGORY"].iloc[idx], frequent_icd9category)
            ),
            "icd10code": torch.from_numpy(
                labeltarget(self.df["ICD10"].iloc[idx], frequent_icd10code)
            ),
            "icd10cat": torch.from_numpy(
                labeltarget(self.df["ICD10_CATEGORY"].iloc[idx], frequent_icd10category)
            ),
        }

        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        }
        return resp, target
