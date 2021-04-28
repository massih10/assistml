#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import h2o
import time
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from sklearn.utils import resample

h2o.init()


# data_train_all = h2o.import_file("/home/sruthi/asm-2/asm-2/1_data/Reviews_Upsampling_train_50.csv")
# data_train_all['Score'] = data_train_all['Score'].asfactor()
# data_test_all=h2o.import_file("/home/sruthi/asm-2/asm-2/1_data/Reviews_Upsampling_test_50.csv")
# data_test_all['Score'] = data_test_all['Score'].asfactor()

h2o.init()
df_clean=pd.read_csv('/home/sruthi/asm-2/asm-2/1_data/baseline.csv')
df_clean=resample(df_clean,replace=True,stratify=df_clean['ProductId'])
#df = h2o.import_file("/home/sruthi/asm-2/asm-2/1_data/baseline.csv")
df =h2o.H2OFrame(df_clean)
df['Score'] = df['Score'].asfactor()

metrics={}

# Fetch the factors and the output
y = 'Score'
x = df.col_names
x.remove(y)
x.remove('ProductId')
train, test = df.split_frame(ratios=[.5])


model_review = H2ODeepLearningEstimator(hidden = [200,200,200,200],balance_classes = True,epochs = 10,nfolds = 5, fold_assignment = 'Stratified',l1 = 0.01, mini_batch_size = 100, input_dropout_ratio = 0.1, seed = 50, standardize = True)
start=time.time()
model_review.train(x=x,y=y, training_frame=train, model_id="DLN_afr_002")
end=time.time()
t=(end-start)

modelfile = model_review.download_mojo(path="/home/sruthi/asm-2/asm-2/3_pickle", get_genmodel_jar=True)
print("Model saved to " + modelfile)




cnf = model_review.model_performance(test_data=test).confusion_matrix().as_data_frame().as_matrix().tolist()

t=(end-start)
print((model_review.model_performance(test_data=test).confusion_matrix()))
print(cnf)
accuracy = 1-model_review.model_performance(test_data=test).confusion_matrix().as_data_frame().as_matrix().tolist()[5][5]
#precision list
for i in range(5):
    if cnf[5][i] == 0:
        cnf[5][i] = 1        
p=[cnf[0][0]/cnf[5][0],cnf[1][1]/cnf[5][1],cnf[2][2]/cnf[5][2],cnf[3][3]/cnf[5][3],cnf[4][4]/cnf[5][4]]
r=[1-cnf[0][5],1-cnf[1][5],1-cnf[2][5],1-cnf[3][5],1-cnf[4][5]]
print("accuracy: ",accuracy)
print("Training time: ",t)    
print("precision: ",p)  
print("recall: ",r)  
metrics['accuracy']=accuracy
metrics['error']=1-accuracy
metrics['precision']=p
metrics['recall']=r
metrics['confusion_matrix_rowstrue_colspred']=cnf
metrics["test_time_per_unit"] = t/299408
metrics['training_time']=t
metrics['cross_validated_training_time']=t
metrics['test_file']='reviews_upsampled_test_50.csv'

import os,json
if os.path.exists('/home/sruthi/asm-2/asm-2/3_pickle/DLN_afr_002.json'):
    with open('/home/sruthi/asm-2/asm-2/3_pickle/DLN_afr_002.json', 'r') as f:
        models = json.load(f)
    models["Metrics"] = metrics
    with open('/home/sruthi/asm-2/asm-2/3_pickle/DLN_afr_002.json', 'w') as f:
        json.dump(models, f, indent = 2) 







h2o.cluster().shutdown()
