'''
Created on 13 Apr 2018
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from util.classifier_util import save_decision_tree, display_scores

def logistic_regression(X_train, y_train, X_test, y_test, class_list, classifier_type="binary"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        path = "../models/logitreg_binary.pkl"
    else:
        path = "../models/logitreg_multiclass.pkl"
    joblib.dump(model, path) 
    # get the error of training set
    print("")
    print("Logistic Regression")
    display_scores(model, X_train, y_train, X_test, y_test, class_list)
    return model
    
def decision_tree(X_train, y_train, X_test, y_test, class_list, classifier_type="binary"):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        path = "../models/tree_binary.pkl"
        
    else:
        path = "../models/tree_multiclass.pkl"
    joblib.dump(model, path)
    print("")
    print("Decision Tree")
    display_scores(model, X_train, y_train, X_test, y_test, class_list)
    print("Visualising tree")
    path = save_decision_tree(model, classifier_type, class_list)
    print("Graph saved in ", path) 
    return model

def linear_svc(X_train, y_train, X_test, y_test, class_list, classifier_type="binary"):
    model = LinearSVC(loss="hinge")
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        path = "../models/svc_binary.pkl"
    else:
        path = "../models/svc_multiclass.pkl"
    joblib.dump(model, path) 
    # get the error of training set
    print("")
    print("SVC")
    display_scores(model, X_train, y_train, X_test, y_test, class_list)
    return model
    
def mlp_classifier(X_train, y_train, X_test, y_test, class_list, classifier_type="binary"):
    model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,), activation="logistic", max_iter=600)
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        path = "../models/mlp_binary.pkl"
    else:
        path = "../models/mlp_multiclass.pkl"
    joblib.dump(model, path) 
    # get the error of training set
    print("")
    print("MLP Classifier")
    display_scores(model, X_train, y_train, X_test, y_test, class_list)
    return model