#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Created on Tue May 26 01:07:34 2020

@author: sruthi
"""

import pandas as pd
import h2o
import time
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix

h2o.init()


df = h2o.import_file("/home/sruthi/asm-2/asm-2/1_data/bank.csv")
df['Class'] = df['Class'].asfactor()
df['V2'] = df['V2'].asfactor()
df['V3'] = df['V3'].asfactor()
df['V4'] = df['V4'].asfactor()
df['V5'] = df['V5'].asfactor()
df['V7'] = df['V7'].asfactor()
df['V8'] = df['V8'].asfactor()
df['V9'] = df['V9'].asfactor()
df['V11'] = df['V11'].asfactor()
df['V16'] = df['V16'].asfactor()
y = 'Class'
x = df.col_names
x.remove(y)

train, test = df.split_frame(ratios=[.75])

model_bank = H2OGradientBoostingEstimator(ntrees = 50, seed = 25,nfolds=5, max_depth=10,balance_classes=True)   ## Instantiating the class



start=time.time()
model_bank.train(x=x, y=y, training_frame=train)
end=time.time()
    
t=(end-start)

modelfile = model_bank.download_mojo(path="/home/sruthi/asm-2/asm-2/3_pickle", get_genmodel_jar=True)
print("Model saved to " + modelfile)

pred = model_bank.predict(test)

#data_train_all = pd.read_csv("/home/sruthi/asm-2/asm-2/1_data/bank_upsampled_train_75.csv")
# data_test_all=pd.read_csv("/home/sruthi/asm-2/asm-2/1_data/bank_upsampled_test_25.csv")

# data_train_all=pd.get_dummies(data_train_all,drop_first=False)
# data_test_all=pd.get_dummies(data_test_all,drop_first=False)

# data_train_h2o=h2o.H2OFrame(data_train_all)
# data_test_h2o=h2o.H2OFrame(data_test_all)

# data_train_h2o['Class']=data_train_h2o['Class'].asfactor()

# model_bank = H2OGradientBoostingEstimator(ntrees = 50, seed = 25,nfolds=5, max_depth=5)   ## Instantiating the class

# model_bank.train(x=data_train_h2o.names[1:],y=data_train_h2o.names[0], training_frame=data_train_h2o, model_id="GBM_bank",
#             validation_frame=data_train_h2o)

# print(model_bank.cross_validation_metrics_summary())

# # perf = model.model_performance()
# # perf.mean_score()
# x=data_train_h2o.names[1:]
# perf = model_bank.model_performance()
metrics = {}
a = model_bank.model_performance(test_data=test).confusion_matrix().to_list()[0][0]
b= model_bank.model_performance(test_data=test).confusion_matrix().to_list()[0][1]
c = model_bank.model_performance(test_data=test).confusion_matrix().to_list()[1][0]
d = model_bank.model_performance(test_data=test).confusion_matrix().to_list()[1][1]
recall = d / (c+d)
precision = d/(b+d)
f1 = 2*(precision * recall)/(precision + recall)
accuracy = (a+d)/(a+b+c+d)
metrics = {}
metrics["Accuracy"]=accuracy
metrics["Error"]=1-accuracy
metrics["Precision"]=precision
metrics["Recall"]=recall
metrics["FScore"]=f1
metrics["Single_training_time"] = t
metrics["Cross_validated_Training_time"] = t
metrics["Test_time_per_unit"] = t/11303
metrics["Confusion_Matrix_rowstrue_colspred"] = [a,b,c,d]
metrics["Test_File"] = "bank_upsampled_test_25.csv"
print("accuracy: ",accuracy)
print("error,",1-accuracy)
print("recall",recall)
print("precision",precision)
print("f1",f1)
print("Training time: ",t)    

import os,json
if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/GBE_bank_003.json'):
    with open('/home/sruthi/asm-2/asm-2/3_pickle/GBE_bank_003.json', 'r') as f:
        models = json.load(f)
    models["Metrics"] = metrics
    with open('/home/sruthi/asm-2/asm-2/3_pickle/GBE_bank_003.json', 'w') as f:
        json.dump(models, f, indent = 2) 


# h2o.cluster().shutdown()