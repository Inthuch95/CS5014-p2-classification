'''
Created on 13 Apr 2018
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.regression import mean_squared_error
import pydotplus
import numpy as np

def logistic_regression(X_train, y_train, X_test, y_test, classifier_type="binary"):
    X_train = np.c_[np.ones_like(X_train), X_train]
    X_test = np.c_[np.ones_like(X_test), X_test]
    
    print("Training model")
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    if classifier_type == "binary":
        joblib.dump(model, "../models/logitreg_binary.pkl")
    else:
        joblib.dump(model, "../models/logitreg_multiclass.pkl")
        
    # get the error of training set
    print("")
    train_predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, train_predictions)
    rmse = np.sqrt(mse)
    print("Train set result")
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R^2: ", model.score(X_train, y_train))
    
    # get the error of testing set
    print("")
    test_predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)
    print("Test set result")
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R^2: ", model.score(X_test, y_test))
    return model
    
def decision_tree(X_train, y_train, X_test, y_test, classifier_type="binary"):
    print("Training model")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    class_list = []
    if classifier_type == "binary":
        class_list = ["Red", "Green"]
        joblib.dump(model, "../models/tree_binary.pkl")
    else:
        joblib.dump(model, "../models/tree_multiclass.pkl")
        class_list = ["Blue", "Green", "Pink", "Red", "Yellow"]
    
    print("Visualising tree")
    dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=["X["+str(i)+"]" for i in range(X_train.shape[1])],
            class_names=class_list,
            rounded=True,
            filled=True,
            impurity=False
        )
    graph = pydotplus.graph_from_dot_data(dot_data)
    if classifier_type == "binary":
        graph.write_pdf("../models/tree_binary.pdf")
    else:
        graph.write_pdf("../models/tree_multiclass.pdf")
    print("Graph saved") 
    return model