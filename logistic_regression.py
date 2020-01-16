"""======================================= Logistic - Feature Selection =================================="""
data_bycycle_final_vars = data_bycycle_final.columns.values.tolist()
print(data_bycycle_final_vars)
lg_Y=['Status']
lg_X=[i for i in data_bycycle_final_vars if i not in lg_Y ]
print(lg_X)
print(lg_Y)

# carryout feature selection
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model)
rfe = rfe.fit(data_bycycle_final[lg_X],data_bycycle_final[lg_Y] )
#print(rfe.support_)
#print(rfe.ranking_)

rank = rfe.ranking_

length = len(lg_X)
len(rank)

# Update X and Y with selected features
cols = []
for i in range(length):
    print(lg_X[i], " rank is : ", rank[i])
    if rank[i] == 1:
        cols.append(lg_X[i]) 

print(lg_X)
print(cols)

"""========================= Logistic - X, y & Devide into test, train ========================="""
lg_X = data_bycycle_final[cols]
lg_Y = data_bycycle_final['Status']

print(lg_Y)
labels = lg_Y.unique()
print(labels)

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import numpy as np
# setting up testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(lg_X, lg_Y, test_size=0.25, random_state=27)

"""========================= Logistic - Before dealing with imbalanced set ==================="""
clf1 = linear_model.LogisticRegression(solver='liblinear')
clf1.fit(X_train,  Y_train)
#3- Run the test data against the new model
predicted = clf1.predict(X_test)
#4-Check model accuracy
print("Logistic - Original, accuracy_score: ", metrics.accuracy_score(Y_test, predicted))	
print("Logistic - Original, f1_score: ", metrics.f1_score(Y_test, predicted, average='weighted', labels=np.unique(predicted)))
print("Logistic - Original, precision_score: ", metrics.precision_score(Y_test, predicted, average='weighted'))
print("Logistic - Original, recall_score: ", metrics.recall_score(Y_test, predicted, average='weighted'))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, predicted, labels))

"""============================== Logistic - Oversample =============================="""
from sklearn.utils import resample
# concatenate our training data back together
lg_X = pd.concat([X_train, Y_train], axis=1)
# separate minority and majority classes
stolen = lg_X[lg_X.Status==0]
recovered = lg_X[lg_X.Status==1]

lg_X.Status.value_counts()
# upsample minority
recovered_upsampled = resample(recovered,
                          replace=True, # sample with replacement
                          n_samples=len(stolen), # match number in majority class
                          random_state=27) # reproducible results
# combine majority and upsampled minority
upsampled = pd.concat([stolen, recovered_upsampled])
# check new class counts
upsampled.Status.value_counts()
# trying logistic regression again with the balanced dataset
new_Y_train = upsampled.Status
new_X_train = upsampled.drop('Status', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(new_X_train, new_Y_train)

lg_upsampled_pred = upsampled.predict(X_test)

# Checking accuracy
print("Logistic - Oversample, accuracy_score: ", metrics.accuracy_score(Y_test, lg_upsampled_pred))	
print("Logistic - Oversample, f1_score: ", metrics.f1_score(Y_test, lg_upsampled_pred, average='weighted', labels=np.unique(predicted)))
print("Logistic - Oversample, precision_score: ", metrics.precision_score(Y_test, lg_upsampled_pred, average='weighted'))
print("Logistic - Oversample, recall_score: ", metrics.recall_score(Y_test, lg_upsampled_pred, average='weighted'))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, lg_upsampled_pred, labels))

"""============================== Logistic - Undersample =============================="""
# downsample majority
stolen_downsampled = resample(stolen,
                                replace = False, # sample without replacement
                                n_samples = len(recovered), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([stolen_downsampled, recovered])

# checking counts
downsampled.Status.value_counts()

# trying logistic regression again with the undersampled dataset
downsampled_Y_train = downsampled.Status
downsampled_X_train = downsampled.drop('Status', axis=1)

undersampled = LogisticRegression(solver='liblinear').fit(downsampled_X_train, downsampled_Y_train)

undersampled_pred = undersampled.predict(X_test)

# Checking accuracy
print("Logistic - Undersample, accuracy_score: ", metrics.accuracy_score(Y_test, undersampled_pred))	
print("Logistic - Undersample, f1_score: ", metrics.f1_score(Y_test, undersampled_pred, average='weighted', labels=np.unique(predicted)))
print("Logistic - Undersample, precision_score: ", metrics.precision_score(Y_test, undersampled_pred, average='weighted'))
print("Logistic - Undersample, recall_score: ", metrics.recall_score(Y_test, undersampled_pred, average='weighted'))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, undersampled_pred, labels))

"""============================== Logistic - synthetic samples =============================="""
from imblearn.over_sampling import SMOTE

lg_sm = SMOTE(random_state=27, ratio=1.0)
lg_sm_X_train, lg_sm_Y_train = lg_sm.fit_sample(X_train, Y_train)

lg_smote = LogisticRegression(solver='liblinear').fit(lg_sm_X_train, lg_sm_Y_train)

lg_smote_pred = lg_smote.predict(X_test)

# Checking accuracy
print("Logistic - synthetic, accuracy_score: ", metrics.accuracy_score(Y_test, lg_smote_pred))	
print("Logistic - synthetic, f1_score: ", metrics.f1_score(Y_test, lg_smote_pred, average='weighted', labels=np.unique(predicted)))
print("Logistic - synthetic, precision_score: ", metrics.precision_score(Y_test, lg_smote_pred, average='weighted'))
print("Logistic - synthetic, recall_score: ", metrics.recall_score(Y_test, lg_smote_pred, average='weighted'))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, lg_smote_pred, labels))