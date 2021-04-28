import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix

missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df = pd.read_csv('/home/sruthi/asm-2/1_data/bank.csv',na_values = missing_values)
# df = df.drop(labels=['V10','V13'],axis=1)
# np.random.seed(25)
print(len(df))

seed = 55
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
    ('classifier', LinearSVC(penalty='l2',loss='squared_hinge', dual=True, tol=0.0001, C=0.01, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=seed, max_iter=3000))])


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
filename = '/home/sruthi/asm-2/asm-2/3_pickle/SVM_bank_005.pkl'
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
c = confusion_matrix(y_test, y_pred)
d = c.tolist()
p,r,f,sp=prec_recall
t=(stop-start)
t1=(stop1-start1)
t2=(stop3-start3)
s=accuracy_score(y_test,y_pred)
metrics['Accuracy']=s
metrics['Error']=1-s
metrics['Precision']=p[1]
metrics['Recall']=r[1]
metrics['FScore']=f[1]
# metrics['Precision']=p
# metrics['Recall']=r
# metrics['FScore']=f
metrics["Cross_validated_Training_time"] = t1
metrics["Test_time_per_unit"] = t2/11303
metrics["Confusion_Matrix_rowstrue_colspred"] = d
metrics["Test_File"] = "bank_stratified_onehot_test_25.csv"

print("Training time: ",t)    
print("Score= {}".format(float(s)))
print("Error= {}".format(1-s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print(confusion_matrix(y_test,y_pred))


# import os,json
# if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/SVM_bank_005.json'):
#     with open('/home/sruthi/asm-2/asm-2/3_pickle/SVM_bank_005.json', 'r') as f:
#         models = json.load(f)
#     models["Metrics"] = metrics
#     with open('/home/sruthi/asm-2/asm-2/3_pickle/SVM_bank_005.json', 'w') as f:
#         json.dump(models, f, indent = 2)
   
pl = preprocessing.named_transformers_['cat']
ohe = pl.named_steps['onehotencoding']
fn = ohe.get_feature_names()
print((fn).tolist())
print(len(fn))



                                    

