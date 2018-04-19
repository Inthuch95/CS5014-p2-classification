'''
Created on Apr 16, 2018
'''
from algorithms.classifiers import logistic_regression, decision_tree, mlp_classifier
from util.classifier_util import load_train_test
import pandas as pd
import numpy as np
    
classifier_type = "binary"

if classifier_type == "binary":
    path = "../binary/"
else:
    path = "../multiclass/"
wavelength_path = path + "Wavelength.csv" 
wavelength = pd.read_csv(wavelength_path, header=None)
wavelength_list = []
for wl in wavelength[0]:
    name = "wl_" + str(wl)
    wavelength_list.append(name)
with open(path+"key.txt",'r') as f:
    class_dict = eval(f.read())
class_list = list(class_dict.values())

X_train, y_train, X_test, y_test = load_train_test(classifier_type)
# X_train = np.reshape(X_train, (-1,1))
# X_test = np.reshape(X_test, (-1,1))

logist_clf = logistic_regression(X_train, y_train, X_test, y_test, class_list, classifier_type)
tree_clf = decision_tree(X_train, y_train, X_test, y_test, class_list, classifier_type)
mlp_clf = mlp_classifier(X_train, y_train, X_test, y_test, class_list, classifier_type)