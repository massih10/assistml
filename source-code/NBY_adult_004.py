# !/usr/bin/env python
# coding: utf-8
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
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score, \
    confusion_matrix
from sklearn.naive_bayes import BernoulliNB

missing_values = ["n/a", "na", "--", "NA", "?", "", " ", -1, "NAN", "NaN"]
df = pd.read_csv('/home/osama/HiWi/new-version/asm-2/1_data/adult_1590.csv', na_values=missing_values)

# df = df.drop(labels=['PurchDate','Nationality','Make','Model','AUCGUART','PRIMEUNIT','Trim','VNZIP1','VNST','BYRNO','SubModel','VehicleAge','VehOdo','WarrantyCost'],axis=1)
# df = df.dropna(axis=0)
print(len(df))
y = df.pop('class')
X = df

print(df.shape)

categorical_columns = df.select_dtypes(exclude=['int', 'float']).columns
numerical_columns = df.select_dtypes(include=['int', 'float']).columns

numerical_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median'))
])

categorical_pipe = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

# print(len(categorical_pipe.columns))

preprocess = ColumnTransformer(
    transformers=[('cat', categorical_pipe, categorical_columns), ('num', numerical_pipe, numerical_columns)])

bnb = Pipeline(steps=[
    ('preprocess', preprocess),
    ('classifier', BernoulliNB())])

seed = 25
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=seed)
start = time.time()
bnb.fit(X_train, y_train)
stop = time.time()
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
start1 = time.time()
score = cross_val_score(bnb, X_test, y_test, cv=5, scoring='accuracy')
stop1 = time.time()
start3 = time.time()
y_pred = bnb.predict(X_test)
stop3 = time.time()
accuracy = bnb.score(X_test, y_test)

print(f"Training_Time_in_s_before_cv: {stop - start}s")
print(f"Training_Time_in_s_for_cv: {stop1 - start1}s")

import joblib

filename = '/home/osama/HiWi/new-version/asm-2/3_pickle/NBY_adult_004.pkl'
with open(filename, 'wb') as file:
    joblib.dump(bnb, file)
with open(filename, 'rb') as file:
    dt_pkl = joblib.load(file)

y_pred = dt_pkl.predict(X_test)
# accuracy = dt_pkl.score(X_test, y_test)
# print("Accuracy:",accuracy)
# print("Error:",1 - accuracy)
# print("Precision:",precision_score(y_test, y_pred, average='weighted'))
# print("Recall:",recall_score(y_test, y_pred, average='weighted'))
# print("FScore:",f1_score(y_test, y_pred, average='weighted'))
metrics = {}
prec_recall = precision_recall_fscore_support(y_test, y_pred, average=None)
p, r, f, sp = prec_recall
t = (stop - start)
t1 = (stop1 - start1)
t2 = (stop3 - start3)
s = accuracy_score(y_test, y_pred)
metrics['Accuracy'] = s
metrics['Error'] = 1 - s
metrics['Precision'] = p[0]
metrics['Recall'] = r[0]
metrics['FScore'] = f[0]
metrics["single_training_time"] = t
metrics["cross_validated_training_time"] = t1
metrics['test_time_per_unit'] = t2 / 11302
metrics['test_file'] ="adult_test_25_25.csv"
metrics['confusion_matrix_rowstrue_colspred'] = confusion_matrix(y_test,y_pred).tolist()
print("Score= {}".format(float(s)))
print("Error= {}".format(1 - s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print("test time per unit", (t2 / 11302))
# confusion= ('Confusion_Matrix',y_test,y_pred)
print(confusion_matrix(y_test, y_pred))

pl = preprocess.named_transformers_['cat']
ohe = pl.named_steps['onehot']
print(ohe.get_feature_names())
print(len(ohe.get_feature_names()))

import os
import json
if os.path.exists('/home/osama/HiWi/new-version/asm-2/3_pickle/NBY_adult_004.json'):
    with open('/home/osama/HiWi/new-version/asm-2/3_pickle/NBY_adult_004.json', 'r') as infile:
        models = json.load(infile)
    models["Model"]["Metrics"] = metrics
    with open('/home/osama/HiWi/new-version/asm-2/3_pickle/NBY_adult_004.json', 'w') as infile:
        json.dump(models, infile, indent=2)




