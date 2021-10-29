#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:25:09 2021

@author: sehunii
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


df = pd.read_excel(r'/Users/sehunii/Spyder/default_of_credit_card_clients.xls')

#clean the dataframe
df = df.iloc[1:,1:]
features = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13',
            'X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']
x = df.loc[:, features].values
y = df.loc[:,['Y']].values


# split the data
train_data, test_data, train_lbl, test_lbl = train_test_split(x,y,test_size=0.3, 
                                             random_state = 1922)
train_lbl = train_lbl.astype('int').ravel()
test_lbl = test_lbl.astype('int').ravel()

#standardize the data
std_train = StandardScaler().fit_transform(train_data)
std_test = StandardScaler().fit_transform(test_data)


#choose the minimum number of principal components st. 95% of the variance is retained
pca = PCA(.95)
pca.fit(std_train)
trans_train = pca.transform(std_train)
trans_test = pca.transform(std_test)

#number of principal components is 15
print(trans_train.shape[1])

# fit logistic regression model
start = time.perf_counter()
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(trans_train, train_lbl)
end = time.perf_counter()
print(end-start)

# confusion matric training set
train_result = logisticRegr.predict(trans_train)
score_train = round(logisticRegr.score(trans_train, train_lbl),4)
print(score_train)
cm1 = metrics.confusion_matrix(train_lbl, train_result)
print(cm1)

# predictions
predictions = logisticRegr.predict(trans_test)
score = round(logisticRegr.score(trans_test, test_lbl),4)
print(score)

# confusion matrix
cm2 = metrics.confusion_matrix(test_lbl, predictions)
print(cm2)

#try to produce a projection plot
#first two principal components
principalComponents = pca.fit_transform(std_train)
firstTwo = principalComponents[:,:2]
principalDf = pd.DataFrame(data = firstTwo,
              columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Y']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Y'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


#plot variance explained

pca_all = PCA()
pca_all.fit(std_train)
exp_var_pca = pca_all.explained_variance_ratio_
exp_var_cumul = np.cumsum(exp_var_pca)

plt.plot(exp_var_cumul)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# variance explained bar chart
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.show()


