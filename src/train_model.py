'''
Created on Apr 16, 2018
'''
from algorithms.classifiers import logistic_regression, decision_tree
from util.load_data import load_train_test

classifier_type = "multiclass"
print("Loading data")
X_train, y_train, X_test, y_test = load_train_test(classifier_type)
model = logistic_regression(X_train, y_train, X_test, y_test, classifier_type)
# model = decision_tree(X_train, y_train, X_test, y_test, classifier_type)