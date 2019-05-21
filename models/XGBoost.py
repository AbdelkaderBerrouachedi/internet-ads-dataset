import pandas as pd
import numpy as np
import os

# Import names
f = open('../data/dictionary'.format(os.getcwd()),'r')
columns_string = f.read()
columns = columns_string.split(',')

# Import the dataset
dataset = pd.read_csv('../data/ad.data'.format(os.getcwd()),
                      header=None,
                      names=columns,
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

# Feature reduction
# 1. chi2 - squared distribution
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

number_of_features = 50

# Feature selection
test = SelectKBest(score_func=chi2,k=number_of_features)
fit = test.fit(X,y)

# Summarize scores
np.set_printoptions(precision=3)
scores = fit.scores_

features = fit.transform(X)
# Summarize selected features

def rank_features(scores,k):
    max_indices = []
    a = list(scores)
    b = list(scores)
    for i in range(0, k):
        max_indices.append(b.index(np.max(a)))
        a.pop(a.index(np.max(a)))
    return max_indices

print("Chi2 :",rank_features(scores,number_of_features))

# Feature scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(features)

# Split data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1,1])/np.sum(cm)
perc_accuracy = accuracy * 100

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()