'''
Created on Apr 16, 2018
'''
from sklearn.decomposition import PCA
from algorithms.classifiers import logistic_regression, decision_tree, mlp_classifier
from util.classifier_util import load_train_test
import pandas as pd
    
classifier_type = "binary"
use_pca = False

if classifier_type == "binary":
    path = "../binary/"
else:
    path = "../multiclass/"
wavelength_path = path + "Wavelength.csv" 
wavelength = pd.read_csv(wavelength_path, header=None)
wavelength_list = []
for wl in wavelength[0]:
    name = "wavelength_" + str(wl)
    wavelength_list.append(name)
with open(path+"key.txt",'r') as f:
    class_dict = eval(f.read())
class_list = list(class_dict.values())

X_train, y_train, X_test, y_test = load_train_test(classifier_type)
if use_pca:
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

logist_clf = logistic_regression(X_train, y_train, X_test, y_test, class_list, classifier_type)
tree_clf = decision_tree(X_train, y_train, X_test, y_test, class_list, classifier_type)
mlp_clf = mlp_classifier(X_train, y_train, X_test, y_test, class_list, classifier_type)