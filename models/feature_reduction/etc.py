import pandas as pd
import numpy as np
import os

# Import the dataset
dataset = pd.read_csv(
    '{}/data/ad.data'.format(os.getcwd()),
    header=None,
    encoding='utf-8',
    low_memory=False)

# Handle missing data
dataset = dataset.replace('?',np.nan)
dataset = dataset.replace('   ?',np.nan)
dataset = dataset.replace('     ?',np.nan)

# Replace null values with mode value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
dataset.iloc[:,0:4] = imputer.fit_transform(dataset.iloc[:,0:4])

# Label encode the y attribute
# 0 - ad.
# 1 - nonad.
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset.iloc[:,1558] = label_encoder.fit_transform(dataset.iloc[:,1558])

# Describe the dataset
dataset_describe = dataset.describe()

# Feature extraction
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)

number_of_features = 60

def rank_features(scores,k):
    max_indices = []
    a = list(scores)
    b = list(scores)
    for i in range(0, k):
        max_indices.append(b.index(np.max(a)))
        a.pop(a.index(np.max(a)))
    return max_indices

print(rank_features(model.feature_importances_,number_of_features))