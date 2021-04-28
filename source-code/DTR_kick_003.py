#!/usr/bin/env python
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
from sklearn.metrics import precision_recall_fscore_support,f1_score,precision_score,recall_score,accuracy_score,confusion_matrix


#np.random.seed(42)
missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df = pd.read_csv('/home/sruthi/asm-2/1_data/kick.csv',na_values = missing_values)

# df = df.drop(labels=['PurchDate','Nationality','Make','Model','AUCGUART','PRIMEUNIT','Trim','VNZIP1','VNST','BYRNO','SubModel','VehicleAge','VehOdo','WarrantyCost'],axis=1)
df = df.drop(labels=['AUCGUART','PRIMEUNIT'],axis=1)
df.fillna(df.median())
df = df.dropna(axis=0)
print(df.shape)
print(len(df))
df = pd.get_dummies(df)
print(list(df.columns.values))
# y = df.pop('IsBadBuy')
# X = df
# print(df.shape)

# categorical_columns = ['Color','WheelTypeID','Auction','Transmission','WheelType','Size','TopThreeAmericanName','IsOnlineSale']

# numerical_columns = ['VehYear',
#  'MMRAcquisitionAuctionAveragePrice',
#  'MMRAcquisitionAuctionCleanPrice',
#  'MMRAcquisitionRetailAveragePrice',
#  'MMRAcquisitonRetailCleanPrice',
#  'MMRCurrentAuctionAveragePrice',
#  'MMRCurrentAuctionCleanPrice',
#  'MMRCurrentRetailAveragePrice',
#  'MMRCurrentRetailCleanPrice',
#  'VehBCost']

# numerical_columns = [key for key in dict(df.dtypes)
#                    if dict(df.dtypes)[key]
#                        in ['float64','float32','int32','int64']]

# numerical_pipe = Pipeline(steps=[
#     ('impute', SimpleImputer(strategy='median'))
# ])

# #print(len(categorical_pipe.columns))

# preprocess = ColumnTransformer(transformers=[('num', numerical_pipe, numerical_columns)])

# dt = Pipeline(steps=[
#     ('preprocess', preprocess),
#     ('classifier', DecisionTreeClassifier(random_state=0))])

# seed = 25
# X_train, X_test, y_train, y_test = train_test_split(X, y , stratify=y, test_size=0.25,random_state = seed)
# start = time.time()
# dt.fit(X_train,y_train)
# stop = time.time()
# #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# start1 = time.time()
# score = cross_val_score(dt, X_test, y_test, cv=5,scoring='accuracy')
# stop1 = time.time()
# start3 = time.time()
# y_pred = dt.predict(X_test)
# stop3 = time.time()
# accuracy = dt.score(X_test, y_test)

# print(f"Training_Time_in_s_before_cv: {stop - start}s")
# print(f"Training_Time_in_s_for_cv: {stop1 - start1}s")

# import joblib
# filename = '/home/sruthi/asm-2/asm-2/3_pickle/DTR_kick_003.pkl'
# with open(filename, 'wb') as file:
#     joblib.dump(dt, file)
# with open(filename, 'rb') as file:
#     dt_pkl = joblib.load(file)

# y_pred = dt_pkl.predict(X_test)
# # accuracy = dt_pkl.score(X_test, y_test)
# # print("Accuracy:",accuracy)
# # print("Error:",1 - accuracy)
# # print("Precision:",precision_score(y_test, y_pred, average='weighted'))
# # print("Recall:",recall_score(y_test, y_pred, average='weighted'))
# # print("FScore:",f1_score(y_test, y_pred, average='weighted'))
# metrics={}
# prec_recall=precision_recall_fscore_support(y_test,y_pred,average=None)
# p,r,f,sp=prec_recall
# t=(stop-start)
# t1=(stop1-start1)
# t2=(stop3-start3)
# s=accuracy_score(y_test,y_pred)
# metrics['Accuracy']=s
# metrics['Error']=1-s
# metrics['Precision']=p
# metrics['Recall']=r
# metrics['FScore']=f
# print("Score= {}".format(float(s)))
# print("Error= {}".format(1-s))
# print("Precision,Recall,F_beta,Support {}".format(prec_recall))
# print("test time per unit",(t2/18245))
# print(confusion_matrix(y_test,y_pred))


# #dt['preprocess'].transformers_[1][1]['onehot']\
#     #.get_feature_names(categorical_columns)
    
# #dt.named_steps['preprocess'].transformers_[1][1]\
#    #.named_steps['onehot'].get_feature_names(categorical_columns)
   
# # pl = preprocess.named_transformers_['cat']
# # ohe = pl.named_steps['onehot']
# # print(ohe.get_feature_names())
# # print(len(ohe.get_feature_names()))

# # pl1 = preprocess.named_transformers_['num']
# # imp = pl1.named_steps['impute']
# # print(imp.get_feature_names())
# # print(len(imp.get_feature_names()))


                                    