from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def generate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
