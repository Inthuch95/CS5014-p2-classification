'''
Created on Apr 16, 2018
'''
from algorithms.classifiers import logistic_regression, decision_tree, mlp_classifier, linear_svc
from util.classifier_util import load_train_test
# import numpy as np

classifier_type = "binary"    
# classifier_type = "multiclass"
if classifier_type == "binary":
    path = "../binary/"
else:
    path = "../multiclass/"
with open(path+"key.txt",'r') as f:
    class_dict = eval(f.read())
class_list = list(class_dict.values())

X_train, y_train, X_test, y_test = load_train_test(classifier_type)

# logist_clf = logistic_regression(X_train, y_train, X_test, y_test, class_list, classifier_type)
# tree_clf = decision_tree(X_train, y_train, X_test, y_test, class_list, classifier_type)
# mlp_clf = mlp_classifier(X_train, y_train, X_test, y_test, class_list, classifier_type)
svc_clf = linear_svc(X_train, y_train, X_test, y_test, class_list, classifier_type)