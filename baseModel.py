#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:34:37 2021

@author: sehunii
"""
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# original data logistic regression
start_all = time.perf_counter()
logisticRegr_all = LogisticRegression(solver='lbfgs')
logisticRegr_all.fit(train_data, train_lbl)
end_all = time.perf_counter()
print(end_all-start_all)

# confusion matric training set
result = logisticRegr_all.predict(train_data)
score_train_all = round(logisticRegr_all.score(train_data, train_lbl),4)
print(score_train_all)
print(metrics.confusion_matrix(train_lbl, result))

predictions_all = logisticRegr_all.predict(test_data)
score_all = round(logisticRegr_all.score(test_data, test_lbl),4)
print(score_all)

print(metrics.confusion_matrix(test_lbl, predictions_all))