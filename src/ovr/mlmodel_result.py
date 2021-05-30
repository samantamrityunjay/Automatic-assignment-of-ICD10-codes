from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score,precision_score

def mlmodel_result(y_test, y_predicted):
    print("Accuracy Score: ",accuracy_score(y_test, y_predicted))
    print("Hamming loss:", hamming_loss(y_test, y_predicted))
    print("Precision: ",precision_score(y_test, y_predicted,average='micro'))
    print("Recall: ", recall_score(y_test, y_predicted,average='micro'))
    print("F1_score:", f1_score(y_test, y_predicted,average='micro'))