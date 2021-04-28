import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import resample
import time
from sklearn.naive_bayes import BernoulliNB, ComplementNB, CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score, \
    confusion_matrix
from sklearn.feature_selection import VarianceThreshold

# class ModifiedLabelEncoder(LabelEncoder):

#     def fit_transform(self, y, *args, **kwargs):
#         return super().fit_transform(y).reshape(1, 1)

#     def transform(self, y, *args, **kwargs):
#         return super().transform(y).reshape(1, 1)


missing_values = ["n/a", "na", "--", "NA", "?", "", " ", -1, "NAN", "NaN"]
df = pd.read_csv('/home/mohammoa/gasdrift.csv', na_values=missing_values)
df = df.fillna(df.median())
print(df.shape)




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
#v1 = df.V2
#histogram = np.histogram(v1)
#_ = plt.hist(v1, bins=109)  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()

#data = df.drop(['V18','V19','V20','V33','V39','V41','V49','V57','V36'], axis=1)

#print(data)


anova_filter = SelectKBest(f_classif, k=70)
dt = GaussianNB()
anova_svm = make_pipeline(anova_filter, dt)
seed = 25
y = df.Class
X = df.drop('Class', axis=1)

print(X.describe())

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.25, random_state=seed)
# Separate majority and minority classes
# X = pd.concat([X_train, y_train], axis=1)
# df_majority = X[X.IsBadBuy==0]
# df_minority = X[X.IsBadBuy==1]
# # Upsample minority class
# df_minority_upsampled = resample(df_minority,
#                                  replace=True,     # sample with replacement
#                                  n_samples=len(df_majority),    # to match majority class
#                                  random_state=seed) # reproducible results

# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# y_train = df_upsampled.IsBadBuy
# X_train = df_upsampled.drop('IsBadBuy', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.25, random_state=seed, shuffle=True)

start = time.time()
anova_svm.fit(X_train, y_train)

stop = time.time()
print(X_test)
y_pred = anova_svm.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
start1 = time.time()
score = cross_val_score(anova_svm, X_test, y_test, cv=5, scoring='accuracy')
stop1 = time.time()
print(f"Training_Time_in_s_before_cv: {stop - start}s")
print(f"Training_Time_in_s_for_cv: {stop1 - start1}s")

import joblib

filename = '/home/mohammoa/HiWi/asm-2/3_pickle/NBY_gasdrift_001.pkl'
with open(filename, 'wb') as file:
    joblib.dump(anova_svm, file)
with open(filename, 'rb') as file:
    dt_pkl = joblib.load(file)

start3 = time.time()
y_pred = dt_pkl.predict(X_test)
stop3 = time.time()
metrics = {}
prec_recall = precision_recall_fscore_support(y_test, y_pred, average='weighted')
p, r, f, sp = prec_recall
t = (stop - start)
t1 = (stop1 - start1)
t2 = (stop3 - start3)
s = accuracy_score(y_test, y_pred)
c = confusion_matrix(y_test, y_pred)
d = c.tolist()
metrics['accuracy'] = s
metrics['error'] = 1 - s
metrics['precision'] = p
metrics['recall'] = r
metrics['fscore'] = f
metrics["single_training_time"] = t
metrics["cross_validated_training_time"] = t1
metrics['test_time_per_unit'] = t2 / 17358
metrics["confusion_matrix_rowstrue_colspred"] = d
metrics['test_file'] = "gasdrift_test_25.csv"
print("Score= {}".format(s))
print("Error= {}".format(1 - s))
print("Precision,Recall,F_beta,Support {}".format(prec_recall))
print(confusion_matrix(y_test, y_pred))
print("test time per unit", (t2 / 17358))

import os, json

if os.path.exists('/home/mohammoa/HiWi/asm-2/3_pickle/NBY_gasdrift_001.json'):
    with open('/home/mohammoa/HiWi/asm-2/3_pickle/NBY_gasdrift_001.json', 'r') as f:
        models = json.load(f)
    models["NBY_gasdrift_001"]["Metrics"] = metrics
    with open('/home/mohammoa/HiWi/asm-2/3_pickle/NBY_gasdrift_001.json', 'w') as f:
        json.dump(models, f, indent=1)