#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import time
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import precision_recall_fscore_support,f1_score,precision_score,recall_score,accuracy_score,confusion_matrix

# class ModifiedLabelEncoder(LabelEncoder):

#     def fit_transform(self, y, *args, **kwargs):
#         return super().fit_transform(y).reshape(1, 1)

#     def transform(self, y, *args, **kwargs):
#         return super().transform(y).reshape(1, 1)


missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df = pd.read_csv('/home/sruthi/asm-2/1_data/kick.csv',na_values=missing_values)
df = df.drop(labels=['AUCGUART','PRIMEUNIT','Trim','VNZIP1','VNST','BYRNO','SubModel','WheelTypeID','VehYear','Color'],axis=1)
df = df.fillna(df.median())
df = df.dropna(subset=['Transmission','WheelType','Nationality','Size','TopThreeAmericanName'])
df= pd.get_dummies(df)

print(df.shape)
# np.random.seed(25)
#Pipeline approach starts here
#categorical_columns = ['Auction','Make','Model','IsOnlineSale','Transmission','WheelType','Nationality','Size','TopThreeAmericanName']

# numerical_columns = ['PurchDate', 'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice','VehBCost','WarrantyCost']

# numerical_columns = []

# numerical_pipe = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
# ])

# # categorical_pipe = Pipeline([
# #     ('onehotencoding', OneHotEncoder())])


# # preprocessing_pipeline = ColumnTransformer(
# #     [('cat', categorical_pipe, categorical_columns),
# #      ('num', numerical_pipe, numerical_columns)])

# preprocessing_pipeline = ColumnTransformer(
#     [('num', numerical_pipe, numerical_columns)])
# dt = Pipeline([
#     ('preprocess', preprocessing_pipeline),
#     ('classifier', DecisionTreeClassifier(random_state=0,min_impurity_decrease=0.001))])
 

#End of pipeline code

dt = ExtraTreesClassifier(n_estimators=5,max_depth=5)
seed = 25
y = df.IsBadBuy
X = df.drop('IsBadBuy', axis=1)




df_majority = df[df['IsBadBuy']==0]
df_minority = df[df['IsBadBuy']==1]
df_minority_upsampled = resample(df_minority, replace=True,n_samples=len(df_majority),random_state=seed)     # sample with replacement
df=pd.concat([df_majority, df_minority_upsampled])

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['IsBadBuy'] ,stratify=df['IsBadBuy'],test_size=0.25,random_state=seed,shuffle=True)



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
filename = '/home/sruthi/asm-2/3_pickle/RFR_kick_007.pkl'
with open(filename, 'wb') as file:
    joblib.dump(dt, file)
with open(filename, 'rb') as file:
    dt_pkl = joblib.load(file)

start3=time.time()    
y_pred = dt_pkl.predict(X_test)
stop3=time.time()
metrics={}
prec_recall=precision_recall_fscore_support(y_test,y_pred,average='micro')
p,r,f,sp=prec_recall
t=(stop-start)
t1=(stop1-start1)
t2=(stop3-start3)
s=accuracy_score(y_test,y_pred)
c = confusion_matrix(y_test,y_pred)
d = c.tolist()
metrics['Accuracy']=s
metrics['Error']=1-s
metrics['Precision']=p
metrics['Recall']=r
metrics['FScore']=f
metrics["Single_training_time"] = t
metrics["Cross_validated_Training_time"] = t1
metrics['Test_time_per_unit'] = t2/17358
metrics["Confusion_Matrix_rowstrue_colspred"] = d
metrics['Test_File'] = "kick_upsampled_onehot_test_25.csv"
print("Score= {}".format(s))
print("Error= {}".format(1-s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print(confusion_matrix(y_test,y_pred))
print("test time per unit",(t2/17358))

import os,json

if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/RFR_kick_007.json'):
    with open('/home/sruthi/asm-2/asm-2/3_pickle/RFR_kick_007.json', 'r') as f:
        models = json.load(f)
    models["Metrics"] = metrics
    with open('/home/sruthi/asm-2/asm-2/3_pickle/RFR_kick_007.json', 'w') as f:
        json.dump(models, f, indent = 1)