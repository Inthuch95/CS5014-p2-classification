'''
Created on Apr 21, 2018
'''
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Load data
print("Loading data")
X_binary = np.loadtxt("../prepared_data/binary/XToClassify.txt")
X_multiclass = np.loadtxt("../prepared_data/multiclass/XToClassify.txt")

# load models
print("Loading SVC models")
binary_clf = joblib.load("../models/svc_binary.pkl")
multiclass_clf = joblib.load("../models/svc_multiclass.pkl")

#predict the classes
print("Making predictions")
binary_predictions = binary_clf.predict(X_binary)
multiclass_predictions = multiclass_clf.predict(X_multiclass)

# save results
print("Saving predictions to files")
binary_df = pd.DataFrame(binary_predictions)
binary_df.to_csv("../binaryTask/PredictedClasses.csv", index=False)
multiclass_df = pd.DataFrame(multiclass_predictions)
multiclass_df.to_csv("../multiClassTask/PredictedClasses.csv", index=False)
print("Predicted classes saved")