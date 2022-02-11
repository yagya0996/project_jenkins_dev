#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 08:14:33 2021

@author: yagyadatta
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('Sonar.csv')

df_dummy = pd.get_dummies(df,drop_first=True)


X = df_dummy.drop('Class_R',axis=1)
y = df_dummy['Class_R']

kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)


parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'random_state':[2021],'C':np.linspace(0.001,40,25)}

logreg = LogisticRegression()

cv = GridSearchCV(logreg, param_grid=parameters,cv=kfold,scoring='roc_auc')

cv.fit(X, y)
cv_df=pd.DataFrame(cv.cv_results_)
print(cv.best_params_)
print(cv.best_score_)
print(cv.best_estimator_)



"""
{'penalty': 'l2', 'random_state': 2021, 'solver': 'newton-cg'}
0.849218847514042
LogisticRegression(random_state=2021, solver='newton-cg')
"""

"""
{'C': 1.667625, 'penalty': 'l2', 'random_state': 2021, 'solver': 'saga'}
0.8505533596837944
LogisticRegression(C=1.667625, random_state=2021, solver='saga')
"""