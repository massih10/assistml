import pickle
import time
import pandas as pd
from joblib import dump, load

# Imports required for dataset fetching and processing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# Imports required for measuring Classsifer's performance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# Get gasdrift dataset fom openml
dataset_gas_drift = fetch_openml(data_id=1476, as_frame=True, return_X_y=True)
print("Gas Drift Dataset before Feature Scaling:: ")
print(dataset_gas_drift)

# X --> features, y --> label 
X = dataset_gas_drift[0]
Y = dataset_gas_drift[1] 

# Data Preprocessing --> Feature Scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

print("\nDataset after Feature Scaling ::")
print(X)
print(Y)

# dividing X, y into train and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, shuffle=True, random_state = 18)
pd.DataFrame(X_train, Y_train).to_csv('gasdrift_train_40.csv')
pd.DataFrame(X_test, Y_test).to_csv('gasdrift_test_60.csv')

print("\nCompleted Training and Test Dataset splitting process ::") 
print(X_train)
print(Y_train)


# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(
    criterion="mse",
    n_estimators=10,
    learning_rate=0.3,
    random_state=18
)


# Cross validation
start = time.time()
scores = cross_val_score(gbc, X, Y, cv=3, scoring="accuracy")
stop = time.time()
cross_validation_time_secs = stop - start
print(f"Cross Validation time: {cross_validation_time_secs} seconds")
print('3 Fold Cross Validation Scores: {}'.format(scores))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Model Training
start = time.time()
gbc.fit(X_train, Y_train)
stop = time.time()
training_time_secs = stop - start
print(f"Training time: {training_time_secs} seconds")
#Y_pred = gbc.predict(X_test)


# Save Trained Classifier to a file in the current working directory
pkl_filename = "GBE_gasdrift_001.pkl"
with open(pkl_filename, 'wb') as file:
    dump(gbc, file) 
print("\nPickle file stored in local directory")

# Load Trained Classifer from pickle file
with open(pkl_filename, 'rb') as file:
    pickle_model = load(file)
print("\nPickle file loaded successfully from local directory")

Y_pred = pickle_model.predict(X_test)
print("\nNumber of mislabeled points out of total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum())) 
score = pickle_model.score(X_test, Y_test)
print("\nTest score of pickled model: {0:.2f} %".format(100 * score))


# Parameters to estimate Classifier's classification capability
precision, recall, fBeta, support = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
confusion_mat = confusion_matrix(Y_test, Y_pred)

print('Accuracy: {}'.format(scores.mean()))
print('Error: {}'.format(1 - scores.mean()))
print('Precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fBeta: {}'.format(fBeta))
print('Training Time: {}'.format(training_time_secs))
print('Cross Validated Training time: {}'.format(cross_validation_time_secs))


print(confusion_mat.tolist())

