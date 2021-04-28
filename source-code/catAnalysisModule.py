
from scipy import stats
import pandas as pd
import json
import sys



#Chi-square Test of Independence using scipy.stats.chi2_contingency, https://pythonfordatascience.org/chi-square-test-of-independence-python/
#The H0 (Null Hypothesis): There is no relationship between variable one and variable two.
#We can reject the null hypothesis as the p-value is less than 0.05
def corr_cal(df, col_name):
    crosstab = pd.crosstab(df[sys.argv[2]], df[col_name])
    res = stats.chi2_contingency(crosstab)
    p = res[1]
    #return p-value, We can reject the null hypothesis as the p-value is less than 0.05
    if p < 0.05:
      ifCorr = 'True'
    else:
      ifCorr = 'False'
    return(p,ifCorr)

#here we run ordinal test, we manually annotate each feature with a boolean. 
def ordinal_test(col_name, file_name):
    if file_name == 'kick':
        if col_name in ['Size']:
            ordinal = 'True'
        else:
            ordinal = 'False'
    elif file_name == 'bank':
        if col_name in ['V11','V4','V16']:
            ordinal = 'True'
        else:
            ordinal = 'False'
    return ordinal
            
#Number of categories/levels with pandas
def freq_counts(df, col_name):
    res = df[col_name].value_counts()
    val_list = res.tolist()
    index_list = res.index.tolist()        
    return index_list,val_list, len(index_list)

#here we test if the feature is ordered or not with pandas
def order_test(df, col_name):
    return(df[col_name].astype('category').cat.ordered)

#here we How big is the imbalance of the feature (ratio between most popular and least popular)
def imbalance_test(df, col_name):
    res = df[col_name].value_counts().tolist()
    return str(max(res)/min(res))

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


#load dataset, handle_missing_values, and generate json
def preprocess():   
    #Get the dataset file name from input argument
    file_name = sys.argv[1]
    #load dataset
    missing_values = ["n/a", "na", "--","NA","?",""," ","NAN","NaN"]
    df_in=pd.read_csv('./'+file_name+'/'+file_name+'.csv',na_values=missing_values)
    
    #handle_missing_values
    (df,miss_value,drop_cols)=handle_missing_values(df_in)
    #get all cat columns
    num_col = map(str, sys.argv[3].strip('[]').split(','))
    #content of Json file
    metrics = {}

    #for each num_col, calculate Quantiles, outlier, and distribution. Then store as JSON
    for col in num_col:
        if(col not in drop_cols):
            #calculate the correlation between the col and the target column.
            (p, ifCorr)= corr_cal(df, col)
            correlarion = {'p-value':p,'ifCorrelated':ifCorr,'significance':'0.05'}
            freq_counts(df, col)
            #number of columns
            (index_list,val_list,num_levels) = freq_counts(df, col)
            levels = {}
            for i in range(len(val_list)):
                levels[str(index_list[i])] = str(val_list[i])
            #ordinal test
            ordinal = ordinal_test(col, file_name)
            #imbalance_test
            ratio = imbalance_test(df, col)
            #missing values
            miss_val = str(miss_value[col])
            #create json content
            metrics[col]={'correlation':correlarion,'levels':levels, 'imbalence':ratio, 'missing values':miss_val, 'ordinal':ordinal,'num_levels':num_levels}
        else:
            print(col+" is dropped for having missing values more than 1/4 the whole size of the dataset")

    #write content to json     
    meta_data = {file_name:metrics}    
    make_json(meta_data, "cat_"+file_name)
    return df



def run_my_models():
    df =preprocess()

    


run_my_models()  
