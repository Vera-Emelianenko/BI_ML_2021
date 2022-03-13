import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    y_true = y_true.astype(int)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range (len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_pred[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_pred[i] == 1 and y_true[i] == 0: 
                fp += 1
            elif y_pred[i] == 0 and y_true[i] == 1: 
                fn += 1
    if tp+fp != 0: 
        precision = tp/(tp+fp)
    else: 
        precision = 'NA'

    if tp+fn != 0: 
        recall = tp/(tp+fn)
    else: 
        recall = 'NA'

    accuracy = (tp+tn)/(tp+tn+fn+fp)
    if isinstance(precision+recall, float): 
        if precision+recall != 0: 
            f1 = 2*precision*recall/(precision+recall)
    return {'precision':precision, 'recall':recall, 'accuracy':accuracy, 'f1':f1, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    pass
    