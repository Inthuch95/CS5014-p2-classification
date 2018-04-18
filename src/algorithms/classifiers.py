'''
Created on 13 Apr 2018
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from util.classifier_util import save_decision_tree, display_scores

def logistic_regression(X_train, y_train, X_test, y_test, classifier_type="binary"):
    print("Training model")
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        path = "../models/logitreg_binary.pkl"
    else:
        path = "../models/logitreg_multiclass.pkl"
    joblib.dump(model, path)
    print("Model saved in ", path)    
    # get the error of training set
    print("")
    print("Logistic Regression")
    display_scores(model, X_train, y_train, X_test, y_test)
    return model
    
def decision_tree(X_train, y_train, X_test, y_test, classifier_type="binary"):
    print("Training model")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        path = "../models/tree_binary.pkl"
        
    else:
        path = "../models/tree_multiclass.pkl"
    joblib.dump(model, path)
    print("Model saved in ", path)
    print("")
    print("Decision Tree")
    display_scores(model, X_train, y_train, X_test, y_test)
#     print("Visualising tree")
#     feature_list = ["X["+str(i)+"]" for i in range(X_train.shape[1])]
#     path = save_decision_tree(model, classifier_type, feature_list)
#     print("Graph saved in ", path) 
    return model