'''
Created on Apr 16, 2018
'''
from sklearn.decomposition import PCA
from algorithms.classifiers import logistic_regression, decision_tree
from util.classifier_util import load_train_test

classifier_type = "multiclass"
use_pca = False
print("Loading data")
X_train, y_train, X_test, y_test = load_train_test(classifier_type)
if use_pca:
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

model = logistic_regression(X_train, y_train, X_test, y_test, classifier_type)
model = decision_tree(X_train, y_train, X_test, y_test, classifier_type)