import pandas as pd
import h2o
import time
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import numpy as np

h2o.init()


data_train_all = h2o.import_file("/home/sruthi/asm-2/asm-2/1_data/kick_upsampled_train_75.csv")
data_train_all['IsBadBuy'] = data_train_all['IsBadBuy'].asfactor()
data_train_all['WheelTypeID'] = data_train_all['WheelTypeID'].asfactor()
data_train_all['IsOnlineSale'] = data_train_all['IsOnlineSale'].asfactor()
data_test_all=h2o.import_file("/home/sruthi/asm-2/asm-2/1_data/kick_upsampled_test_25.csv")
data_test_all['IsBadBuy'] = data_test_all['IsBadBuy'].asfactor()
data_test_all['WheelTypeID'] = data_test_all['WheelTypeID'].asfactor()
data_test_all['IsOnlineSale'] = data_test_all['IsOnlineSale'].asfactor()

y = 'IsBadBuy'
x = data_train_all.col_names
x.remove(y)

model_bank = H2ODeepLearningEstimator(hidden = [50,50,50,50,50],balance_classes = True,epochs = 10,nfolds = 5, fold_assignment = 'Stratified',l1 = 0.01, mini_batch_size = 100, input_dropout_ratio = 0.1, seed = 25, standardize = True)
start=time.time()
model_bank.train(x=x,y=y, training_frame=data_train_all, model_id="DLN_kick_010")
end=time.time()
t=(end-start)    

modelfile = model_bank.download_mojo(path="/home/sruthi/asm-2/asm-2/3_pickle", get_genmodel_jar=True)
print("Model saved to " + modelfile)


pred = model_bank.predict(data_test_all)

a = model_bank.model_performance(test_data=data_test_all).confusion_matrix().to_list()[0][0]
b= model_bank.model_performance(test_data=data_test_all).confusion_matrix().to_list()[0][1]
c = model_bank.model_performance(test_data=data_test_all).confusion_matrix().to_list()[1][0]
d = model_bank.model_performance(test_data=data_test_all).confusion_matrix().to_list()[1][1]
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
metrics["Test_time_per_unit"] = t/18248
metrics["Confusion_Matrix_rowstrue_colspred"] = [a,b,c,d]
metrics["Test_File"] = "bank_upsampled_test_25.csv"
print("accuracy: ",accuracy)
print("error,",1-accuracy)
print("recall",recall)
print("precision",precision)
print("f1",f1)
print("Training time: ",t)    

import os,json
if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/DLN_kick_010.json'):
    with open('/home/sruthi/asm-2/asm-2/3_pickle/DLN_kick_010.json', 'r') as f:
        models = json.load(f)
    models["Metrics"] = metrics
    with open('/home/sruthi/asm-2/asm-2/3_pickle/DLN_kick_010.json', 'w') as f:
        json.dump(models, f, indent = 2) 


h2o.cluster().shutdown()