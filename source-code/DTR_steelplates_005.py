#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.utils import resample

missing_values = ["n/a", "na", "--", "NA", "?", "", " ", -1, "NAN", "NaN"]
df = pd.read_csv('/home/osama/HiWi/new-version/asm-2/1_data/steelplates_num_2.csv', na_values=missing_values)

seed = 25
dt = DecisionTreeClassifier(max_depth=15)

df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority),
                                 random_state=seed)  # sample with replacement
df = pd.concat([df_majority, df_minority_upsampled])

y = df.pop('target')
X = df

print(df.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=seed, shuffle=True)
start = time.time()
dt.fit(X_train,y_train)

stop = time.time()

y_pred = dt.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
start1=time.time()
score = cross_val_score(dt, X_test, y_test, cv=5,scoring='accuracy')
stop1 = time.time()
print(f"Training_Time_in_s_before_cv: {stop - start}s")
print(f"Training_Time_in_s_for_cv: {stop1 - start1}s")

import joblib

filename = '/home/osama/HiWi/new-version/asm-2/3_pickle/DTR_steelplates_005.pkl'
with open(filename, 'wb') as file:
    joblib.dump(dt, file)
with open(filename, 'rb') as file:
    dt_pkl = joblib.load(file)

start3 = time.time()
y_pred = dt_pkl.predict(X_test)
stop3 = time.time()
metrics={}
prec_recall=precision_recall_fscore_support(y_test,y_pred,average='micro')
p,r,f,sp=prec_recall
t=(stop-start)
t1=(stop1-start1)
t2 = (stop3 - start3)
s=accuracy_score(y_test,y_pred)
metrics['Accuracy']=s
metrics['Error']=1-s
metrics['Precision']=p
metrics['Recall']=r
metrics['FScore']=f
metrics["single_training_time"] = t
metrics["cross_validated_training_time"] = t1
metrics['test_time_per_unit'] = t2 / 18245
metrics['test_file'] ="steelplates_test_25_25.csv"
metrics['confusion_matrix_rowstrue_colspred'] = confusion_matrix(y_test,y_pred).tolist()
print("Score= {}".format(s))
print("Error= {}".format(1-s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print(confusion_matrix(y_test,y_pred))
print("test time per unit",(t/18245))

import json
import os

if os.path.exists('/home/osama/HiWi/new-version/asm-2/3_pickle/DTR_steelplates_005.json'):
    with open('/home/osama/HiWi/new-version/asm-2/3_pickle/DTR_steelplates_005.json', 'r') as f:
        models = json.load(f)
    models["Model"]["Metrics"] = metrics
    with open('/home/osama/HiWi/new-version/asm-2/3_pickle/DTR_steelplates_005.json', 'w') as f:
        json.dump(models, f, indent=2)