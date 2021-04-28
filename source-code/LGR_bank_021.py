#!/usr/bin/env python
# coding: utf-8
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from sklearn.utils import resample

missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df = pd.read_csv('/home/sruthi/asm-2/1_data/bank.csv',na_values = missing_values)

#df = df.drop(labels=['PurchDate','Nationality','Make','Model','AUCGUART','PRIMEUNIT','Trim','VNZIP1','VNST','BYRNO','SubModel','VehicleAge','VehOdo','WarrantyCost'],axis=1)
#df = df.dropna(axis=0)
print(len(df))


print(df.shape)
y = df.pop('Class')
X = df
categorical_columns = df.select_dtypes(exclude=['int', 'float']).columns
numerical_columns = df.select_dtypes(include=['int', 'float']).columns


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
    ('classifier', LogisticRegression(penalty='l2', dual=False, C=0.1, max_iter=1000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None))])

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
filename = '/home/sruthi/Documents/asm-2/3_pickle/LGR_bank_021.pkl'
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
metrics['Accuracy']=s
metrics['Error']=1-s
metrics['Precision']=p
metrics['Recall']=r
metrics['FScore']=f
print("Score= {}".format(float(s)))
print("Error= {}".format(1-s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print("test time per unit",(t2/11302))
print(confusion_matrix(y_test,y_pred))


#dt['preprocess'].transformers_[1][1]['onehot']\
    #.get_feature_names(categorical_columns)
    
#dt.named_steps['preprocess'].transformers_[1][1]\
   #.named_steps['onehot'].get_feature_names(categorical_columns)
   
pl = preprocessing.named_transformers_['cat']
ohe = pl.named_steps['onehotencoding']
fn = ohe.get_feature_names()
print((fn).tolist())



                                    