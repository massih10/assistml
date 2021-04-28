import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import time
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import precision_recall_fscore_support,f1_score,precision_score,recall_score,accuracy_score,confusion_matrix
missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df = pd.read_csv('/home/sruthi/asm-2/1_data/kick.csv',na_values=missing_values)
df = df.drop(labels=['AUCGUART','PRIMEUNIT','Trim','VNZIP1','VNST','BYRNO','SubModel','WheelTypeID','VehYear','Color'],axis=1)
df = df.fillna(df.median())
df = df.dropna(subset=['Transmission','WheelType','Nationality','Size','TopThreeAmericanName'])
df= pd.get_dummies(df)
print(df.columns.names)
seed = 25
print(df.shape)

        
        

sv =  SVC(C=1.0, kernel='poly', degree=40, gamma='scale', cache_size=2000, coef0=0.001, probability=True, random_state=seed)

y = df.IsBadBuy
X = df.drop('IsBadBuy', axis=1)

df_majority = df[df['IsBadBuy']==0]
df_minority = df[df['IsBadBuy']==1]
df_minority_upsampled = resample(df_minority, replace=True,n_samples=len(df_majority),random_state=seed)     # sample with replacement
df=pd.concat([df_majority, df_minority_upsampled])


# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['IsBadBuy'] ,stratify=df['IsBadBuy'],test_size=0.25,random_state=seed,shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['IsBadBuy'] ,stratify=df['IsBadBuy'],train_size = 0.01,test_size=0.005,shuffle=True)


start = time.time()
sv.fit(X_train,y_train)

stop = time.time()
y_pred = sv.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
start1=time.time()
score = cross_val_score(sv, X_test, y_test, cv=5,scoring='accuracy')
stop1 = time.time()
print(f"Training_Time_in_s_before_cv: {stop - start}s")
print(f"Training_Time_in_s_for_cv: {stop1 - start1}s")


import joblib
filename = '/home/sruthi/asm-2/asm-2/3_pickle/SVM_kick_007.pkl'
with open(filename, 'wb') as file:
    joblib.dump(sv, file)
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
t2 =(stop3-start3)
c = confusion_matrix(y_test, y_pred)
d = c.tolist()
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
#metrics['Precision']=p[1]
#metrics['Recall']=r[1]
#metrics['FScore']=f[1]
#metrics['Precision1']=p[0]
#metrics['Recall1']=r[0]
#metrics['FScore1']=f[0]
# metrics['Precision']=p
# metrics['Recall']=r
# metrics['FScore']=f
metrics["Cross_validated_Training_time"] = t1
metrics["Test_time_per_unit"] = t2/18248
metrics["Confusion_Matrix_rowstrue_colspred"] = d
metrics["Test_File"] = "kick_stratified_onehot_test_25.csv"
print(metrics)
import os,json
if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/SVM_kick_007.json'):
    with open('/home/sruthi/asm-2/asm-2/3_pickle/SVM_kick_007.json', 'r') as f:
        models = json.load(f)
    models["Metrics"] = metrics
    with open('/home/sruthi/asm-2/asm-2/3_pickle/SVM_kick_007.json', 'w') as f:
        json.dump(models, f, indent = 2)
