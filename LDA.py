# -*- coding: utf-8 -*-
"""
MMF1922 Data Science - Dimension Reduction
Linear Discriminant Analysis
"""

import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
from sklearn import metrics

# define dataset
df = pd.read_excel(r'/Users/sehunii/Spyder/default_of_credit_card_clients.xls')

X = df.iloc[1:, 1:24].values
y = df.iloc[1:, 24].values

# split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=1922)
y_train = y_train.astype('int').ravel()
y_test= y_test.astype('int').ravel()

# define LDA model
model = LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 'auto')

# choose the best model evaluation method
#
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1922)
# grid = dict()
# grid['solver'] = ['svd', 'lsqr', 'eigen']
# search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# search_results = search.fit(X_train, y_train)
# # result
# print('Best Accuracy: %.3f' % search_results.best_score_)
# print('Best Config: %s' % search_results.best_params_)
# # Best Accuracy: 0.814, Config: {'solver': 'svd'}


# choose the best number of components
#
# Create array of explained variance ratios
model.fit(X_train, y_train)
lda_var_ratios = model.explained_variance_ratio_


def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
            
    # Return the number of components
    return n_components

n = select_n_components(lda_var_ratios, 0.95)
print(n)

# build model 
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X_train, y_train)
X_train_trans = lda.transform(X_train)
X_test_trans = lda.transform(X_test)

print('print: %.3f'%X_train_trans.shape[1])
# feature selection

# plot model performance for comparison


# classification
start = time.perf_counter()
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(X_train_trans, y_train)
end = time.perf_counter()
print(end-start)

# confusion matric training set
train_result = logisticRegr.predict(X_train_trans)
score_train = round(logisticRegr.score(X_train_trans, y_train),4)
print(score_train)
cm1 = metrics.confusion_matrix(y_train, train_result)
print(cm1)

#predictions
predictions = logisticRegr.predict(X_test_trans)
score_test = round(logisticRegr.score(X_test_trans, y_test),4)
print(score_test)
#0.8076


# confusion matrix test set
cm2 = metrics.confusion_matrix(y_test, predictions)
print(cm2)

# [[6788  222]
 # [1510  480]]


