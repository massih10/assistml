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
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from sklearn.utils import resample

missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df = pd.read_csv('/home/sruthi/asm-2/1_data/bank.csv',na_values = missing_values)
#df = df.drop(labels=['V10','V13'],axis=1)
np.random.seed(25)
print(len(df))


print(df.shape)
y = df.pop('Class')
X = df
categorical_columns = df.select_dtypes(exclude=['int', 'float']).columns
numerical_columns = df.select_dtypes(include=['int', 'float']).columns
print(len(numerical_columns))

numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_pipe = Pipeline([
    ('onehotencoding', OneHotEncoder())
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])
lr = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', DecisionTreeClassifier())])

seed = 25
X_train, X_test, y_train, y_test = train_test_split(X, y , stratify=y, test_size=0.25,random_state = seed)
start = time.time()
lr.fit(X_train,y_train)
stop = time.time()
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
start1 = time.time()
score = cross_val_score(lr, X_test, y_test, cv=5,scoring='accuracy')
stop1 = time.time()
start3 = time.time()
y_pred = lr.predict(X_test)
stop3 = time.time()
accuracy = lr.score(X_test, y_test)

print(f"Training_Time_in_s_before_cv: {stop - start}s")
print(f"Training_Time_in_s_for_cv: {stop1 - start1}s")

import joblib
filename = '/home/sruthi/asm-2/asm-2/3_pickle/DTR_bank_003.pkl'
with open(filename, 'wb') as file:
    joblib.dump(lr, file)
with open(filename, 'rb') as file:
    dt_pkl = joblib.load(file)

y_pred = dt_pkl.predict(X_test)
# accuracy = dt_pkl.score(X_test, y_test)
# print("Accuracy:",accuracy)
# print("Error:",1 - accuracy)
# print("Precision:",precision_score(y_test, y_pred, average='weighted'))
# print("Recall:",recall_score(y_test, y_pred, average='weighted'))
# print("FScore:",f1_score(y_test, y_pred, average='weighted'))
metrics={}
prec_recall=precision_recall_fscore_support(y_test,y_pred,average=None)
p,r,f,sp=prec_recall
t=(stop-start)
t1=(stop1-start1)
t2=(stop3-start3)
s=accuracy_score(y_test,y_pred)
c = confusion_matrix(y_test,y_pred)
d = c.tolist()
metrics['Accuracy']=s
metrics['Error']=1-s
metrics['Precision']=p[0]
metrics['Recall']=r[0]
metrics['FScore']=f[0]
metrics["Single_training_time"] = t
metrics["Cross_validated_Training_time"] = t1
metrics["Test_time_per_unit"] = t2/11303
metrics["Confusion_Matrix_rowstrue_colspred"] = d
metrics["Test_File"] = "kick_stratified_onehot_test_25.csv"
print("Score= {}".format(s))
print("Error= {}".format(1-s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print(confusion_matrix(y_test,y_pred))
print("test time per unit",(t2/11303))

import json
import os

if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/DTR_bank_003.json'):
    with open('/home/sruthi/asm-2/asm-2/3_pickle/DTR_bank_003.json', 'r') as f:
        models = json.load(f)
    models["Metrics"] = metrics
    with open('/home/sruthi/asm-2/asm-2/3_pickle/DTR_bank_003.json', 'w') as f:
        json.dump(models, f, indent = 2)


#dt['preprocess'].transformers_[1][1]['onehot']\
    #.get_feature_names(categorical_columns)
    
#dt.named_steps['preprocess'].transformers_[1][1]\
   #.named_steps['onehot'].get_feature_names(categorical_columns)
   
pl = preprocessing.named_transformers_['cat']
ohe = pl.named_steps['onehotencoding']
fn = ohe.get_feature_names()
print((fn).tolist())
print(len(fn))



                                    

