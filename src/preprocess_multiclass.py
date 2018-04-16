'''
Created on Apr 16, 2018
'''
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from util.df_selector import DataFrameSelector
import pandas as pd
import numpy as np

df = pd.read_csv('../multiclass/X.csv', header=None)
y = pd.read_csv('../multiclass/y.csv', header=None)
df['y'] = y

# split training set and testing set (80/20 ratio)
print("Splitting train/test set")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["y"]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
print(train_set["y"].value_counts() / len(train_set))
X_train = train_set.drop("y", axis=1)
y_train = train_set["y"]
X_test = test_set.drop("y", axis=1)
y_test = test_set["y"]

# create transformation pipeline
# min-max scaler for neural network
full_pipeline = Pipeline([
        ("selector", DataFrameSelector(list(X_train))),
        ("min_max_scaler", MinMaxScaler()),
    ])
print("Transforming inputs")
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)
# save pre-processed data into text files
print("Saving data")
np.savetxt("../prepared_data/multiclass/X_train.txt", X_train)
np.savetxt("../prepared_data/multiclass/y_train.txt", y_train)
np.savetxt("../prepared_data/multiclass/X_test.txt", X_test)
np.savetxt("../prepared_data/multiclass/y_test.txt", y_test)
print("Pre-processing completed")