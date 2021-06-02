# Automatic Assignment of ICD codes

## Introduction
This repo contains codes for assignment of ICD codes to medical/clinical text. Data used here is the MIMICIII dataset. Different models have been tried from linear machine learning models to state of the art pretrained NLP model BERT.

## Structure of the project

At the root of the project, you will have:

- **main.py**: used for training and testing different models
- **requirements.txt**: contains the minimum dependencies for running the project
- **w2vmodel.model**: gensim word2vec model trained on MIMICIII discharge summaries
- **src**: a folder that contains:
  - **bert**: contains utilities and files for pretrained bert model
  - **cnn**: contains utilities and files for CNN model
  - **hybrid**: contains utilities and files for the hybrid model (LSTM+CNN) model
  - **rnn**: contains utilities and files for LSTM and GRU models
  - **ovr**: contains utilities and files for different Machine Learning Models (like LR, SVM, NaiveBayes)
  - **fit.py**: training code for both LSTM and CNN models
  - **test_results.py**: inferencing code for trained model used for both LSTM and CNN models
  - **utils.py**: genearal utility codes used for all the models

## Dependencies
 The dependencies are mentioned in the `requirements.txt` file.
 They can be installed by:
 ```bash
pip install -r requirements.txt
```

## How to use the code

Launch train.py with the following arguments:

- `train_path`: path of the training data. 
- `test_path`: path of the test data
- `model_name`: one of the 5 models implemented ['bert', 'hybrid', 'lstm', 'gru', 'cnn', 'ovr']. Default to 'bert'
- `icd_type`: training on different types of icd labels, ['icd9cat', 'icd9code', 'icd10cat', 'icd10code']. Default to 'icd9cat'
- `epochs`: number of epochs 
- `batch_size`: batch size, default to 16 (for bert model).
- `learning_rate`: default to 2e-5 (for bert model)
- `w2vmodel`: path for pretrained gensim word2vec model.

***Example***
```bash
python main.py --train_path train.csv --test_path test.csv --model_name cnn
```

## Data
The data used for training can be downloaded from:
- [train data](https://drive.google.com/file/d/1--ZVpt614neHN9erxmsg6s6aGInThJ22/view?usp=sharing)
- [test data](https://drive.google.com/file/d/1-4tp0og0I7KyNMoqF2_t1smu0_GqQCVf/view?usp=sharing)

