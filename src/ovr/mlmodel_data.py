from src.utils import labeltarget, preprocess
import numpy as np

frequent_icd9category = ['401','427','276','414','272','250','428','518','285','584']
frequent_icd9code = ['4019', '4280', '42731', '41401', '5849', '25000', '2724', '51881', '5990', '53081']
frequent_icd10category = ['I10', 'I25', 'E78', 'I50', 'I48', 'N17', 'E87', 'E11', 'J96', 'N39']
frequent_icd10code = ['I10', 'I2510', 'I509', 'I4891', 'N179', 'E119', 'E784', 'E785', 'J9690', 'J9600']

def mlmodel_data(df, icdtype):
    X = df['discharge_diagnosis'].values
    y = {}
    y['icd9cat'] = np.array([labeltarget(x, frequent_icd9category) for x in df['ICD9_CATEGORY'].values])
    y['icd9code'] = np.array([labeltarget(x, frequent_icd9category) for x in df['ICD9_CODE'].values])
    y['icd10cat'] = np.array([labeltarget(x, frequent_icd9category) for x in df['ICD10_CATEGORY'].values])
    y['icd10code'] = np.array([labeltarget(x, frequent_icd9category) for x in df['ICD10'].values])
    return X, y[icdtype]
