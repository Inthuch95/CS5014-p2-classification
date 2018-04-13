'''
Created on 13 Apr 2018
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.regression import mean_squared_error
from sklearn.externals import joblib
import numpy as np

print("Loading data")
X_train = np.loadtxt("../prepared_data/X_train.txt")
y_train = np.loadtxt("../prepared_data/y_train.txt")
X_test = np.loadtxt("../prepared_data/X_test.txt")
y_test = np.loadtxt("../prepared_data/y_test.txt")

X_train = np.c_[np.ones_like(X_train), X_train]
X_test = np.c_[np.ones_like(X_test), X_test]

print("Training model")
model = LogisticRegression(n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, "../models/logitreg.pkl")

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