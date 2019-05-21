import pandas as pd
import numpy as np
import os

# Import the dataset
dataset = pd.read_csv(
    '../data/ad.data'.format(os.getcwd()),
    header=None,
    encoding='utf-8',
    low_memory=False)

# Checking the percentage of missing values in each feature/attribute
dataset.isnull().sum() / len(dataset) * 100

# Handle missing data
dataset = dataset.replace('?', np.nan)
dataset = dataset.replace('   ?', np.nan)
dataset = dataset.replace('     ?', np.nan)

# Replace null values with mode value
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataset.iloc[:, 0:4] = imputer.fit_transform(dataset.iloc[:, 0:4])

# Label encode the y attribute
# 0 - ad.
# 1 - nonad.
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
dataset.iloc[:, 1558] = label_encoder.fit_transform(dataset.iloc[:, 1558])

# Describe the dataset
dataset_describe = dataset.describe()

# Feature extraction
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature reduction
# 1. chi2 - squared distribution
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

number_of_features = 30

# Feature selection
test = SelectKBest(score_func=chi2, k=number_of_features)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=3)
scores = fit.scores_

features = fit.transform(X)


# Summarize selected features

def rank_features(scores, k):
    max_indices = []
    a = list(scores)
    b = list(scores)
    for i in range(0, k):
        max_indices.append(b.index(np.max(a)))
        a.pop(a.index(np.max(a)))
    return max_indices


print("Chi2 score function selected attributes:", rank_features(scores, number_of_features))

# Feature scaling
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(features)

# Split data into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit classifier to the training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Confusion matrix evaluation
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
perc_accuracy = accuracy * 100

# K-Fold cross validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

# ------------------------------------------------------------------------------
# Multiple Classifiers
# ------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

clfs = {
    'knn': KNeighborsClassifier(5),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}

f1_scores = dict()
for clf_name in clfs:
    print("=" * 30)
    print("MODEL -", clf_name)
    print("=" * 30)
    clf = clfs[clf_name]
    clf.fit(X, y)

    print("")
    print('**** RESULTS ****')
    print("")
    y_pred = clf.predict(X_test)
    f1_scores[clf_name] = f1_score(y_test, y_pred)
    print("F-score: {:.4%}".format(f1_scores[clf_name]))
    print("")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
    print("Accuracy: {:.4%}".format(accuracy))
    print("")

    print("")
    print('**** CONFUSION MATRIX ****')
    print("")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
