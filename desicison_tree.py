"""=============================== Decision Tree ================================"""
#predictors = cols
print(data_bycycle_final.columns)
predictors = ['Bike_Speed', 'Cost_of_Bike', 
              #'Status', 
              'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter', 
              'Time_Afternoon', 'Time_Evening', 'Time_Morning', 'Time_Night', 
              'City_Etobicoke', 'City_NorthYork', 'City_EastYork', 
              'City_OldToronto', 'City_York', 'City_Scarborough',               
              'Premise_Type_Apartment', 'Premise_Type_Commercial', 
              'Premise_Type_House', 'Premise_Type_NonCommercial', 'Premise_Type_Other', 
              'Premise_Type_Outside', 'Premise_Type_PrivateProperty', 
              'Premise_Type_Station', 'Premise_Type_TrafficSector', 'Bike_Type_BM', 
              'Bike_Type_EL', 'Bike_Type_FO', 'Bike_Type_MT', 'Bike_Type_OT', 
              'Bike_Type_RC', 'Bike_Type_RE', 'Bike_Type_RG', 'Bike_Type_SC', 
              'Bike_Type_TA', 'Bike_Type_TO', 'Bike_Type_TR'
               ]
#print(data_bycycle_final.columns.values.tolist())
target = ['Status']

dt_X = data_bycycle_final[predictors]
print(dt_X)
dt_Y = data_bycycle_final[target]
print(dt_Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dt_X, dt_Y, test_size=0.25, random_state=27)

X_train
X_test

Y_train
Y_train

"""==================== Decision Tree - Before dealing with imbalanced set ==============="""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
dt_original = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=27)
dt_original.fit(X_train,Y_train)

### Test the model using the testing data
testY_predict = dt_original.predict(X_test)
testY_predict.dtype
  
#Import scikit-learn metrics module for accuracy calculation
labels = dt_Y['Status'].unique()
print(labels)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, testY_predict))
print("F1 Score:",metrics.f1_score(Y_test, testY_predict,average='weighted', labels=np.unique(testY_predict)))
print("Precision_score:",metrics.precision_score(Y_test, testY_predict,average='weighted', labels=np.unique(testY_predict)))
print("Recall_score:",metrics.recall_score(Y_test, testY_predict,average='weighted', labels=np.unique(testY_predict)))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, testY_predict, labels))

"""=============================== Decision Tree - Oversample ================================"""
from sklearn.utils import resample

dt_over_X = pd.concat([X_train, Y_train], axis=1)

# separate minority and majority classes
stolen = dt_over_X[dt_over_X.Status==0]
recovered = dt_over_X[dt_over_X.Status==1]

dt_over_X.Status.value_counts()

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
dt_over_trainY = upsampled.Status
dt_over_trainX = upsampled.drop('Status', axis=1)

#upsampled = LogisticRegression(solver='liblinear').fit(trainX, trainY)
from sklearn.tree import DecisionTreeClassifier
dt_upsampled = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=27, max_depth=5)
dt_upsampled.fit(dt_over_trainX, dt_over_trainY)

upsampled_pred = dt_upsampled.predict(X_test)

# Checking accuracy
from sklearn import metrics
print("Decision Tree - Oversample, accuracy_score: ", metrics.accuracy_score(Y_test, upsampled_pred))
print("Decision Tree - Oversample, f1_score: ", metrics.f1_score(Y_test, upsampled_pred))
print("Decision Tree - Oversample, precision_score: ", metrics.precision_score(Y_test, upsampled_pred))
print("Decision Tree - Oversample, recall_score: ", metrics.recall_score(Y_test, upsampled_pred))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, upsampled_pred, labels))

"""=============================== Decision Tree - Undersample ================================"""
from sklearn.utils import resample

dt_under_X = pd.concat([X_train, Y_train], axis=1)

stolen = dt_under_X[dt_under_X.Status==0]
recovered = dt_under_X[dt_under_X.Status==1]

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
dt_down_trainY = downsampled.Status
dt_down_trainX = downsampled.drop('Status', axis=1)

downsampled.Status.value_counts()

#undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier
dt_undersampled = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=27, max_depth=5)
dt_undersampled.fit(dt_down_trainX, dt_down_trainY)

undersampled_pred = dt_undersampled.predict(X_test)

# Checking accuracy
print("Decision Tree - Undersample, accuracy_score: ", metrics.accuracy_score(Y_test, undersampled_pred))
print("Decision Tree - Undersample, f1_score: ", metrics.f1_score(Y_test, undersampled_pred))
print("Decision Tree - Undersample, precision_score: ", metrics.precision_score(Y_test, undersampled_pred))
print("Decision Tree - Undersample, recall_score: ", metrics.recall_score(Y_test, undersampled_pred))

from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, undersampled_pred, labels))

"""=============================== Decision Tree - Synthetic ================================"""
from imblearn.over_sampling import SMOTE
from sklearn import metrics
sm = SMOTE(random_state=27, ratio=1.0)
dt_syn_trainX, dt_syn_trainY = sm.fit_sample(X_train, Y_train)

X_train
print(dt_syn_trainX)

#smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier
dt_smote = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=27, max_depth=5)
dt_smote = dt_smote.fit(dt_syn_trainX, dt_syn_trainY)


smote_pred = dt_smote.predict(X_test)
labels = dt_Y['Status'].unique()
# Checking accuracy
print("Decision Tree - Synthetic, accuracy_score: ", metrics.accuracy_score(Y_test, smote_pred))
print("Decision Tree - Synthetic, f1_score: ", metrics.f1_score(Y_test, smote_pred))
print("Decision Tree - Synthetic, precision_score: ", metrics.precision_score(Y_test, smote_pred))
print("Decision Tree - Synthetic, recall_score: ", metrics.recall_score(Y_test, smote_pred))

#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, smote_pred, labels))