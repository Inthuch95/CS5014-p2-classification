'''
Created on Apr 16, 2018
'''
from sklearn.tree import export_graphviz
from sklearn.metrics.regression import mean_squared_error
# import pydotplus
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

def display_scores(model, X_train, y_train, X_test, y_test):
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

def save_decision_tree(model, classifier_type, feature_list):
    if classifier_type == "binary":
        path = "../models/tree_binary.pdf"
        class_list = ["Red", "Green"]
    else:
        path = "../models/tree_multiclass.pdf"
        class_list = ["Blue", "Green", "Pink", "Red", "Yellow"]
    dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_list,
            class_names=class_list,
            rounded=True,
            filled=True,
            impurity=False
        )
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_pdf(path)
    return path