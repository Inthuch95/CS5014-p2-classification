'''
Created on Apr 16, 2018
'''
from algorithms.classifiers import logistic_regression, decision_tree, mlp_classifier, linear_svc
from util.classifier_util import load_train_test
import sys

classifier_type = sys.argv[1]
algorithm = sys.argv[2]
   
if classifier_type == "-b":
    classifier_type = "binary"
    path = "../binary/"
elif classifier_type == "-m":
    classifier_type = "multiclass"
    path = "../multiclass/"
else:
    print("Invalid parameter")
    quit()

# load class names    
with open(path+"key.txt",'r') as f:
    class_dict = eval(f.read())
class_list = list(class_dict.values())

# load preprocessed data
X_train, y_train, X_test, y_test = load_train_test(classifier_type)

# train the model
if algorithm == "logit":
    logistic_regression(X_train, y_train, X_test, y_test, class_list, classifier_type)
elif algorithm == "tree":
    decision_tree(X_train, y_train, X_test, y_test, class_list, classifier_type)
elif algorithm == "mlp":
    mlp_classifier(X_train, y_train, X_test, y_test, class_list, classifier_type)
elif algorithm == "svc":
    linear_svc(X_train, y_train, X_test, y_test, class_list, classifier_type)
else:
    print("Algorithm unavailable")
    quit()