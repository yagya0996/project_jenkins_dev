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


df = pd.read_csv('/data/Sonar.csv')

df_dummy = pd.get_dummies(df,drop_first=True)


X = df_dummy.drop('Class_R',axis=1)
y = df_dummy['Class_R']

kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)


parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'random_state':[2021],'C':np.linspace(0.01,2,5)}

logreg = LogisticRegression()

cv = GridSearchCV(logreg, param_grid=parameters,cv=kfold,scoring='roc_auc')

cv.fit(X, y)
cv_df=pd.DataFrame(cv.cv_results_)
print(cv.best_params_)
print(cv.best_score_)
print(cv.best_estimator_)

best_model = cv.best_estimator_
y_pred = best_model.predict(X)

df = pd.DataFrame({'y_actual':y,'y_pred':y_pred})

df.to_csv('/output/output.csv', index=False)


