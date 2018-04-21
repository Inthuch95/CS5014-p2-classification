'''
Created on 13 Apr 2018
'''
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys

def select_features(train_set, feature_names, n_features):
    # use logistic regression as an estimator
    model = LogisticRegression()
    rfe = RFE(model, n_features)
    fit = rfe.fit(train_set.drop("y", axis=1), train_set["y"])
    selected_features = []
    for feature, selected in zip(feature_names, fit.support_):
        if selected:
            selected_features.append(feature)
    return selected_features

dataset = sys.argv[1]
if dataset == "-b":
    data_path = "../binary/"
    prepared_path = "../prepared_data/binary/"
elif dataset == "-m":
    data_path = "../multiclass/"
    prepared_path = "../prepared_data/multiclass/"
else:
    print("Invalid parameter")
    quit()

# load the data and assign the headers
wavelength = pd.read_csv(data_path+"Wavelength.csv", header=None)
wavelength_list = []
for wl in wavelength[0]:
    name = "wl_" + str(wl)
    wavelength_list.append(name)
df = pd.read_csv(data_path+'X.csv', header=None, names=wavelength_list)
y = pd.read_csv(data_path+'y.csv', header=None)
XToClassify = pd.read_csv(data_path+'XToClassify.csv', header=None, names=wavelength_list)
df['y'] = y

# split training set and testing set (80/20 ratio)
print("Splitting train/test set")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["y"]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
print(train_set["y"].value_counts() / len(train_set))

# select top 5 features
print("Feature selection")
selected_features = select_features(train_set, wavelength_list, 5)
print("Selected features: ", selected_features)
X_train = train_set[selected_features]
y_train = train_set["y"]
X_test = test_set[selected_features]
y_test = test_set["y"]
XToClassify = XToClassify[selected_features]

# create transformation pipeline
full_pipeline = Pipeline([
        ("scale", StandardScaler()),
    ])
print("Transforming inputs")
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.transform(X_test)
XToClassify = full_pipeline.transform(XToClassify)
print(X_train.shape, X_test.shape)

# save pre-processed data into text files
print("Saving data")
np.savetxt(prepared_path+"X_train.txt", X_train)
np.savetxt(prepared_path+"y_train.txt", y_train)
np.savetxt(prepared_path+"X_test.txt", X_test)
np.savetxt(prepared_path+"y_test.txt", y_test)
np.savetxt(prepared_path+"XToClassify.txt", XToClassify)
with open(prepared_path+"selected_feature.txt", 'w') as f:
    for s in selected_features:
        f.write(s + '\n')
print("Pre-processing completed")