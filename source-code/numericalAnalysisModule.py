from scipy.stats import skewtest

from scipy.stats import kstest 
import pandas as pd
import numpy as np
import json
import sys

def calQuantiles(df, col_name):
    #get 0, .25, .5, 0.75, 1 quantiles
    return(df[col_name].quantile([0, .25, .5, 0.75, 1]))


def detect_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #print("fence_low: "+str(fence_low)+", fence_high: "+str(fence_high))
    df_out = df_in[col_name].loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
    return df_out.to_numpy()

def corr_cal(df, col_name):
    #the corr between the second argument(target col) and current column
    return(df[col_name].corr(df[sys.argv[2]]))

#here we calculate iqr based on (Q3-Q1)/(Q1-Q0)+(Q4-Q3)
def IQR(q_series):
    iqr = (q_series[3]-q_series[1])/((q_series[1]-q_series[0])+(q_series[4]-q_series[3]))
    return(iqr)


#here we do skew test with scipy.stats.skewtest, null hypothesis that the skewness of the population == a corresponding normal distribution.
def skewTest(df, col_name):
    #skew test    
    (k,p) = skewtest(df[col_name].values) 
    if p<0.05:
        distribution = 'skew'
    else:
        distribution = 'normal'
    return(p, distribution)


def make_json(metrics, file_name):    
    with open(file_name +'.json', 'w') as outfile:
        json.dump(metrics, outfile)

#import pandas, handle_missing_values
#return a new dataset after Dropping missing values. And return Number of missing values in each column.
def handle_missing_values(df):
    print('size of df before dropping missing values: '+str(len(df)))
    #print Number of missing values in each column
    print("Number of missing values in each column:")
    miss_value = (df.isnull().sum())
    print(miss_value)
    #print Number of missing values in the whole dataset
    null_stats=miss_value.sum()
    print("Number of missings data in the whole dataset: " + str(null_stats))
    
    #drop the column if this col has missing values more than half of the whole dataset 
    drop_cols=list()
    num_rows=len(df)
    for index,val in enumerate(miss_value):
        if val > num_rows/4:            
            drop_cols.append(df.columns[index])
    print("Dropped columns: "+str(drop_cols))
    df=df.drop(drop_cols,axis=1)


    #Drop the rows even with single NaN or single missing values.
    df = df.dropna()
    print('size of df after dropping missing values: '+str(len(df)))
    return (df,miss_value,drop_cols)    

def preprocess():
    

    #Get the dataset file name from input argument
    file_name = sys.argv[1]
    #load dataset
    missing_values = ["n/a", "na", "--","NA","?",""," ","NAN","NaN"]
    df=pd.read_csv('./'+file_name+'/'+file_name+'.csv',na_values=missing_values)
    
    #handle_missing_values
    (df,miss_value,drop_cols)=handle_missing_values(df)
    
    #get all numeric columns
    num_col = map(str, sys.argv[3].strip('[]').split(','))
    #content of Json file
    metrics = {}

    #for each num_col, calculate Quantiles, outlier, and distribution. Then store as JSON
    for col in num_col:
        if(col not in drop_cols):
            #calcalate missing values
            missing_val = {'number': str(miss_value[col])}
            #calculate Quantiles
            q_series = calQuantiles(df, col).to_numpy()
            iqr = IQR(q_series)
            quantiles={'IQR':iqr,'0th':q_series[0],'0.25th':q_series[1],'0.5th':q_series[2],'0.75th':q_series[3],'1.0th':q_series[4]}
            #calcalate outlier info
            outlier_list = detect_outlier(df, col)
            outlier = {'number':str(len(outlier_list)),'actual values':str(outlier_list)}
            #calcalate p value of skew test
            p,distribution = skewTest(df, col)
            p_value = {'distribution':distribution, 'p_value':p, 'significance':0.05}            
            #calculate the correlation between the col and the target column.
            correlarion = {'value':corr_cal(df, col)}
            #create json content
            metrics[col]={'quantiles': quantiles,'outliers': outlier,'skew_test':p_value,'missing values':missing_val,'correlation':correlarion}
        else:
            print(col+" is dropped for having missing values more than 1/4 the whole size of the dataset")
    #write content to json     
    meta_data = {file_name:metrics}    
    make_json(meta_data, "num_"+file_name)
    return df



def run_my_models():
    df =preprocess()

    


run_my_models()  
