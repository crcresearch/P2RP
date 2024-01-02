# Author: Shengwei You
# Last Edited: 07/30/2023

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import label_binarize
from itertools import cycle


# Load data (remember to update the path name)
data = pd.read_csv('simulation_results.csv')

# Preprocess 'Features' into usable format
data['Features'] = data['Features'].apply(eval)
mlb = MultiLabelBinarizer()
features_encoded = pd.DataFrame(mlb.fit_transform(data.pop('Features')), columns=mlb.classes_, index=data.index)
data = data.join(features_encoded)

# Encode 'Behavior' and 'AccountType' into numerical format
le_behavior = LabelEncoder()
data['Behavior'] = le_behavior.fit_transform(data['Behavior'])

le_account_type = LabelEncoder()
data['AccountType'] = le_account_type.fit_transform(data['AccountType'])

# Split the dataset into train and test
X = data.drop(['Behavior', 'UserID'], axis=1)
y = data['Behavior']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for random forest
param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}

# Initialize the grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')

# Fit the grid search
grid_search.fit(X_train, y_train)




# Binarize the output for ROC calculation
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the best model
best_model.fit(X_train, y_train)

# Compute the prediction
y_score = best_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[i][:, 1])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
