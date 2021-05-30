import pandas as pd
import numpy as np 
import argparse 
from ast import literal_eval 
import torch 
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from gensim.models import Word2Vec

from src.bert.bert_model import BERTclassifier
from src.bert.bert_dataset import BERTdataset
from src.bert.bert_train import bert_fit
from src.bert.bert_utils import bert_test_results

from src.rnn.rnn_utils import count_vocab_index, get_emb_matrix
from src.rnn.rnn_dataset import rnndataset
from src.rnn.lstm import LSTMw2vmodel
from src.rnn.gru import GRUw2vmodel

from src.cnn.cnn_dataset import cnndataset
from src.cnn.cnn import character_cnn

from src.hybrid.hybrid_dataset import hybriddataset
from src.hybrid.hybrid import hybrid
from src.hybrid.hybrid_fit import hybrid_fit
from src.hybrid.hybrid_test_results import hybrid_test_results

from src.ovr.mlmodel_data import mlmodel_data
from src.ovr.mlmodel_result import mlmodel_result
from src.ovr.MLmodels import train_classifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.fit import fit
from src.test_results import test_results

from src.utils import split_indices

def data(args):
    train_diagnosis = pd.read_csv(args.train_path)
    test_diagnosis = pd.read_csv(args.test_path)

    train_diagnosis['ICD9_CODE'] = train_diagnosis['ICD9_CODE'].apply(literal_eval)
    train_diagnosis['ICD9_CATEGORY'] = train_diagnosis['ICD9_CATEGORY'].apply(literal_eval)
    train_diagnosis['ICD10'] = train_diagnosis['ICD10'].apply(literal_eval)
    train_diagnosis['ICD10_CATEGORY'] = train_diagnosis['ICD10_CATEGORY'].apply(literal_eval)

    test_diagnosis['ICD9_CODE'] = test_diagnosis['ICD9_CODE'].apply(literal_eval)
    test_diagnosis['ICD9_CATEGORY'] = test_diagnosis['ICD9_CATEGORY'].apply(literal_eval)
    test_diagnosis['ICD10'] = test_diagnosis['ICD10'].apply(literal_eval)
    test_diagnosis['ICD10_CATEGORY'] = test_diagnosis['ICD10_CATEGORY'].apply(literal_eval)

    return train_diagnosis, test_diagnosis

def run(args):
    
    train_diagnosis,test_diagnosis = data(args)

    SEED = 2021
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if args.model_name == "bert":

        print("Model Name: BERT")

        print("Device: ", device)

        learning_rate = args.learning_rate
        loss_fn = nn.BCELoss()
        opt_fn = torch.optim.Adam

        bert_train_dataset = BERTdataset(train_diagnosis)
        bert_test_dataset = BERTdataset(test_diagnosis)

        train_indices, val_indices = split_indices(bert_train_dataset, validation_split=2/7)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        batch_size = args.batch_size
        bert_train_loader = DataLoader(bert_train_dataset, batch_size = batch_size, sampler=train_sampler)
        bert_val_loader = DataLoader(bert_train_dataset, batch_size = batch_size, sampler=val_sampler)
        bert_test_loader = DataLoader(bert_test_dataset, batch_size= batch_size)

        model = BERTclassifier().to(device)

        print("Batch Size: ", batch_size)
        print("Learning Rate: ", learning_rate)

        bert_fit(args.epochs, model, bert_train_loader, bert_val_loader, args.icd_type, opt_fn, loss_fn, learning_rate, device)
        bert_test_results(model, bert_test_loader, args.icd_type, device)
    

    elif args.model_name == 'gru':
        print("Model Name: gru")
        print("Device: ", device)

        learning_rate = args.learning_rate
        loss_fn = nn.BCELoss()
        opt_fn = torch.optim.Adam

        counts, vocab2index = count_vocab_index(train_diagnosis, test_diagnosis)
        rnn_train_dataset = rnndataset(train_diagnosis, vocab2index)
        rnn_test_dataset = rnndataset(train_diagnosis, vocab2index)

        train_indices, val_indices = split_indices(rnn_train_dataset, validation_split=2/7)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        batch_size = args.batch_size
        rnn_train_loader = DataLoader(rnn_train_dataset, batch_size = batch_size, sampler=train_sampler)
        rnn_val_loader = DataLoader(rnn_train_dataset, batch_size = batch_size, sampler=val_sampler)
        rnn_test_loader = DataLoader(rnn_test_dataset, batch_size= batch_size)
        

        w2vmodel = Word2Vec.load(args.w2vmodel)
        weights = get_emb_matrix(w2vmodel, counts)

        gruw2vmodel = GRUw2vmodel(weights_matrix = weights, hidden_size = 256, num_layers = 2, device = device).to(device)
        
        print("Batch Size: ", batch_size)
        print("Learning Rate: ", learning_rate)

        fit(args.epochs, gruw2vmodel, rnn_train_loader, rnn_val_loader, args.icd_type, opt_fn, loss_fn, learning_rate, device)
        test_results(gruw2vmodel, rnn_test_loader, args.icd_type, device)


    elif args.model_name == 'lstm':
        print("Model Name: lstm")
        print("Device: ", device)

        learning_rate = args.learning_rate
        loss_fn = nn.BCELoss()
        opt_fn = torch.optim.Adam

        counts, vocab2index = count_vocab_index(train_diagnosis, test_diagnosis)
        rnn_train_dataset = rnndataset(train_diagnosis, vocab2index)
        rnn_test_dataset = rnndataset(train_diagnosis, vocab2index)

        train_indices, val_indices = split_indices(rnn_train_dataset, validation_split=2/7)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        batch_size = args.batch_size
        rnn_train_loader = DataLoader(rnn_train_dataset, batch_size = batch_size, sampler=train_sampler)
        rnn_val_loader = DataLoader(rnn_train_dataset, batch_size = batch_size, sampler=val_sampler)
        rnn_test_loader = DataLoader(rnn_test_dataset, batch_size= batch_size)
        

        w2vmodel = Word2Vec.load(args.w2vmodel)
        weights = get_emb_matrix(w2vmodel, counts)

        lstmw2vmodel = LSTMw2vmodel(weights_matrix = weights, hidden_size = 256, num_layers = 2, device = device).to(device)
        
        print("Batch Size: ", batch_size)
        print("Learning Rate: ", learning_rate)

        fit(args.epochs, lstmw2vmodel, rnn_train_loader, rnn_val_loader, args.icd_type, opt_fn, loss_fn, learning_rate, device)
        test_results(lstmw2vmodel, rnn_test_loader, args.icd_type, device)


    elif args.model_name == "cnn":
        print("Model Name: cnn")
        print("Device: ", device)

        learning_rate = args.learning_rate
        loss_fn = nn.BCELoss()
        opt_fn = torch.optim.Adam

        cnn_train_dataset = cnndataset(train_diagnosis)
        cnn_test_dataset = cnndataset(test_diagnosis)

        train_indices, val_indices = split_indices(cnn_train_dataset, validation_split=2/7)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        batch_size = args.batch_size
        cnn_train_loader = DataLoader(cnn_train_dataset, batch_size = batch_size, sampler=train_sampler)
        cnn_val_loader = DataLoader(cnn_train_dataset, batch_size = batch_size, sampler=val_sampler)
        cnn_test_loader = DataLoader(cnn_test_dataset, batch_size= batch_size)

        model = character_cnn(cnn_train_dataset.vocabulary, cnn_train_dataset.sequence_length).to(device)

        print("Batch Size: ", batch_size)
        print("Learning Rate: ", learning_rate)

        fit(args.epochs, model, cnn_train_loader, cnn_val_loader, args.icd_type, opt_fn, loss_fn, learning_rate, device)
        test_results(model, cnn_test_loader, args.icd_type, device)


    elif args.model_name == 'hybrid':
        print("Model Name: hybrid")
        print("Device: ", device)

        learning_rate = args.learning_rate
        loss_fn = nn.BCELoss()
        opt_fn = torch.optim.Adam

        counts, vocab2index = count_vocab_index(train_diagnosis, test_diagnosis)

        hybrid_train_dataset = hybriddataset(train_diagnosis, vocab2index)
        hybrid_test_dataset = hybriddataset(train_diagnosis, vocab2index)

        train_indices, val_indices = split_indices(hybrid_train_dataset, validation_split=2/7)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        batch_size = args.batch_size
        hybrid_train_loader = DataLoader(hybrid_train_dataset, batch_size = batch_size, sampler=train_sampler)
        hybrid_val_loader = DataLoader(hybrid_train_dataset, batch_size = batch_size, sampler=val_sampler)
        hybrid_test_loader = DataLoader(hybrid_test_dataset, batch_size= batch_size)
        

        w2vmodel = Word2Vec.load(args.w2vmodel)
        weights = get_emb_matrix(w2vmodel, counts)

        model = hybrid(hybrid_train_dataset.vocabulary, hybrid_train_dataset.sequence_length, weights_matrix = weights, hidden_size = 256, num_layers = 2).to(device)

        print("Batch Size: ", batch_size)
        print("Learning Rate: ", learning_rate)

        hybrid_fit(args.epochs, model, hybrid_train_loader, hybrid_val_loader, args.icd_type, opt_fn, loss_fn, learning_rate, device)
        hybrid_test_results(model, hybrid_test_loader, args.icd_type, device)

    elif args.model_name == 'ovr':
        print("Model Name: Onevs AllClassifier")
        X_train, y_train = mlmodel_data(train_diagnosis, args.icd_type)
        X_test, y_test = mlmodel_data(test_diagnosis, args.icd_type)

        tfidf_vectorizer = TfidfVectorizer(max_df = 0.8)
        X_train = tfidf_vectorizer.fit_transform(X_train)
        X_test = tfidf_vectorizer.transform(X_test)

        ml_model = train_classifier(X_train, y_train)
        y_predict = ml_model.predict(X_test)

        print('-'*20 + args.icd_type + '-'*20)
        mlmodel_result(y_test, y_predict)







        












if __name__ == "__main__":
    parser = argparse.ArgumentParser("Automatic Assignment of Medical Codes")

    parser.add_argument("--train_path", type = str, default = './data/train.csv')
    parser.add_argument("--test_path", type = str, default = './data/test.csv')

    parser.add_argument("--model_name", type = str, choices = ['bert', 'hybrid', 'gru', 'lstm', 'cnn', 'ovr'], default = "bert")
    parser.add_argument("--icd_type", type = str, choices = ['icd9cat', 'icd9code', 'icd10cat', 'icd10code'], default = 'icd9cat')

    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--learning_rate", type = float, default = 2e-5)
    parser.add_argument("--epochs", type = int, default = 4)

    parser.add_argument("--w2vmodel", type = str, default = "w2vmodel.model")

    args = parser.parse_args()
    run(args)






