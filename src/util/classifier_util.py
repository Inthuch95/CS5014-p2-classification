'''
Created on Apr 16, 2018
'''
from sklearn.tree import export_graphviz
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import itertools
import matplotlib.pyplot as plt
import pydotplus
import numpy as np

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges, class_names=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels(class_names)
    plt.yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

def display_scores(model, X_train, y_train, X_test, y_test, class_list):
    y_train_pred = model.predict(X_train)
    train_confusion = confusion_matrix(y_train, y_train_pred)
    mse = mean_squared_error(y_train, y_train_pred)
    rmse = np.sqrt(mse)
    print("Train set result")
    print("Confusion matrix")
    print(train_confusion)
    print("Cross validation scores (3-fold): " + str(cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")))
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("R^2: " + str(model.score(X_train, y_train)))
    
    # get the error of testing set
    print("")
    y_test_pred = model.predict(X_test)
    test_confusion = confusion_matrix(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    print("Test set result")
    print("Confusion matrix")
    print(test_confusion)
    print("Cross validation scores (3-fold): " + str(cross_val_score(model, X_test, y_test, cv=3, scoring="accuracy")))
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("R^2: " + str(model.score(X_test, y_test)))
    
    np.set_printoptions(precision=1) 
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(train_confusion, class_names=class_list)
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(test_confusion, class_names=class_list)
    plt.show()

def save_decision_tree(model, classifier_type, feature_list, class_list):
    if classifier_type == "binary":
        path = "../models/tree_binary.pdf"
    else:
        path = "../models/tree_multiclass.pdf"
    dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_list,
            class_names=class_list,
            rounded=True,
            filled=True,
            impurity=False
        )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(path)
    return path