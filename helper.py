from sklearn.metrics import accuracy_score, hamming_loss, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import classification_report

def evaluate(target,predicted):
    print("\n" + "*"*20 + " Evaluation Start" + "*"*20 + "\n")
    print(f" Accuracy score : {accuracy_score(target,predicted):.3f}")
    print(f" Hamming Loss : {hamming_loss(target,predicted):.3f}")
    print(f" Recall : {recall_score(target,predicted, average = 'micro'):.3f}")
    print(f" Precision : {precision_score(target,predicted, average = 'micro'):.3f}")
    print(f" F1 score : {precision_score(target,predicted, average = 'micro'):.3f}")
    print("\n" + "*"*20 + " Evaluation End " + "*"*20 + "\n")
