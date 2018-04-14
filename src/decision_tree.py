'''
Created on Apr 14, 2018
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals import joblib
import pydotplus
import numpy as np

print("Loading data")
X_train = np.loadtxt("../prepared_data/X_train.txt")
y_train = np.loadtxt("../prepared_data/y_train.txt")
X_test = np.loadtxt("../prepared_data/X_test.txt")
y_test = np.loadtxt("../prepared_data/y_test.txt")

X_train = np.reshape(X_train, (-1,1))

print("Training model")
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
joblib.dump(tree_clf, "../models/tree.pkl")

print("Visualising tree")
dot_data = export_graphviz(
        tree_clf,
        out_file=None,
        feature_names=["X"],
        class_names=["0","1"],
        rounded=True,
        filled=True,
        impurity=False
    )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("../tree.pdf")
print("Graph saved")