'''
Created on 13 Apr 2018
'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from util.df_selector import DataFrameSelector
import pandas as pd
import numpy as np

df = pd.read_csv('../binary/X.csv', header=None)
y = pd.read_csv('../binary/y.csv', header=None)
df['y'] = y

# split training set and testing set (80/20 ratio)
print("Splitting train/test set")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_set.drop("y", axis=1)
y_train = train_set["y"]
X_test = test_set.drop("y", axis=1)
y_test = test_set["y"]

# create transformation pipeline
full_pipeline = Pipeline([
        ("selector", DataFrameSelector(list(X_train))),
        ("std_scaler", StandardScaler()),
    ])
print("Transforming inputs")
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)

# save pre-processed data into text files
print("Saving data")
np.savetxt("../prepared_data/X_train.txt", X_train)
np.savetxt("../prepared_data/y_train.txt", y_train)
np.savetxt("../prepared_data/X_test.txt", X_test)
np.savetxt("../prepared_data/y_test.txt", y_test)
print("Pre-processing completed")