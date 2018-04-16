'''
Created on Apr 16, 2018
'''
import numpy as np

def load_train_test(dataset):
    if dataset == "binary":
        path = "../prepared_data/binary/"
    else:
        path = "../prepared_data/multiclass/"
    X_train = np.loadtxt(path+"X_train.txt")
    y_train = np.loadtxt(path+"y_train.txt")
    X_test = np.loadtxt(path+"X_test.txt")
    y_test = np.loadtxt(path+"y_test.txt")
    return X_train, y_train, X_test, y_test