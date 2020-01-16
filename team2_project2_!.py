"""
    ==================== Combine those columns into one dataframe ==================== 
    
    Occurrence_Month → Grouping to Season (Weather : Fall, Winter, Summer, Spring)
    Occurrence_Time → Morning, Afternoon, Evening, Night
    Premise_Type 
        Other         3355 → use “Location Type” to categorize more this !
    Bike_Type 
        OT    2987 → Use “Bike_Make” and “Bike_Model” grouping it
        UN       5 → Use “Bike_Make” and “Bike_Model” grouping it à it is clustering!!!!!
    Bike_Speed
        RFE.. column selection !!!!!!!! do feature selection and see drop it or not
    Bike_Colour 
        BLK       5022
        BLU       1650
        GRY       1483
        WHI       1442
        RED       1282
        + Other
    Cost_of_Bike
        Fill NAN replace to MEDIAN!!! → Maximum is too high, Average is misleading
    Status
        - Drop “Unknown” rows !!! 
        - Deal with Imbalance Data
            https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
            1. Switch stolen 0, recovered 1 → replace value !!!!
            2. Get bad data model
            3. Deal with imbalanced data à choose 1 algorithm
    City
        - Using Lat, Long → Categorizing to 6 Cities in Toronto
"""
import pandas as pd
import os

path = "/Users/kyungjin/Downloads/Semester 5/COMP309 - Data Warehouse & Mining HCIS/Project 2"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_team2 = pd.read_csv(fullpath)

"""=========================== Occurrence_Month ==========================="""
data_team2.loc[(data_team2['Occurrence_Month'] < 3),'Season'] = 'Winter'
data_team2.loc[(data_team2['Occurrence_Month'] == 12),'Season'] = 'Winter'
data_team2.loc[(data_team2['Occurrence_Month'] >= 3) & (data_team2['Occurrence_Month'] < 6),'Season'] = 'Spring'
data_team2.loc[(data_team2['Occurrence_Month'] >= 6) & (data_team2['Occurrence_Month'] < 9),'Season'] = 'Summer'
data_team2.loc[(data_team2['Occurrence_Month'] >= 9) & (data_team2['Occurrence_Month'] < 12),'Season'] = 'Fall'

data_team2.Season.value_counts(dropna=False)

"""=========================== Occurrence_Time ==========================="""
data_team2.loc[data_team2['Occurrence_Time'].str.contains('00:'),'Time'] = 'Night'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('01:'),'Time'] = 'Night'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('02:'),'Time'] = 'Night'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('03:'),'Time'] = 'Night'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('04:'),'Time'] = 'Night'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('05:'),'Time'] = 'Night'

data_team2.loc[data_team2['Occurrence_Time'].str.contains('06:'),'Time'] = 'Morning'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('07:'),'Time'] = 'Morning'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('08:'),'Time'] = 'Morning'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('09:'),'Time'] = 'Morning'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('10:'),'Time'] = 'Morning'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('11:'),'Time'] = 'Morning'

data_team2.loc[data_team2['Occurrence_Time'].str.contains('12:'),'Time'] = 'Afternoon'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('13:'),'Time'] = 'Afternoon'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('14:'),'Time'] = 'Afternoon'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('15:'),'Time'] = 'Afternoon'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('16:'),'Time'] = 'Afternoon'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('17:'),'Time'] = 'Afternoon'

data_team2.loc[data_team2['Occurrence_Time'].str.contains('18:'),'Time'] = 'Evening'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('19:'),'Time'] = 'Evening'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('20:'),'Time'] = 'Evening'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('21:'),'Time'] = 'Evening'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('22:'),'Time'] = 'Evening'
data_team2.loc[data_team2['Occurrence_Time'].str.contains('23:'),'Time'] = 'Evening'


data_team2['Time'].value_counts(dropna=False)

"""=========================== Premise_Type ==========================="""

categories = []
for col, col_type in data_team2.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          data_team2[col].fillna(0, inplace=True)

#Create new data set: 
include = ['Occurrence_Day', 'Occurrence_Month',
           'Bike_Type', 'Location_Type','Bike_Colour','Status']
df_ = data_team2[include]
df_ = data_team2[include]

#check unique value of columns in data set df_ :
def unique_col_values(df_):
    for column in df_:
        print("{} | {} | {}".format(
            df_[column].name, len(df_[column].unique()), df_[column].dtype
        ))

#replace null values by NaN:
#df_['Bike_Colour'].replace('-', np.nan, inplace=True)
#drop missing value because the number of missing value is just only 0.08% of data set
#new_df=df_.dropna()
new_df = df_
new_df.isnull().sum()

new_df.Location_Type.value_counts(dropna=False)

#Group Location_type hood in to correlated with Premise_Type
outside=['Streets, Roads, Highways (Bicycle Path, Private Road)','Parking Lots (Apt., Commercial Or Non-Commercial)'
         ,'Open Areas (Lakes, Parks, Rivers)']
house=['Single Home, House (Attach Garage, Cottage, Mobile)','Homeless Shelter / Mission','Retirement Home']
commercial=['Other Commercial / Corporate Places (For Profit, Warehouse, Corp. Bldg','Bank And Other Financial Institutions (Money Mart, Tsx)',
            'Convenience Stores','Commercial Dwelling Unit (Hotel, Motel, B & B, Short Term Rental)',
            'Bar / Restaurant','Construction Site (Warehouse, Trailer, Shed)','Gas Station (Self, Full, Attached Convenience)',
            'Dealership (Car, Motorcycle, Marine, Trailer, Etc.)']
apartment =['Apartment (Rooming House, Condo)']
privateProperty=['Universities / Colleges','Private Property (Pool, Shed, Detached Garage)','Schools During Un-Supervised Activity',
                 'Hospital / Institutions / Medical Facilities (Clinic, Dentist, Morgue)','Other Non Commercial / Corporate Places (Non-Profit, GovT, Firehall)'
                 ,'Police / Courts (Parole Board, Probation Office','Schools During Supervised Activity','Group Homes (Non-Profit, Halfway House, Social Agency)',
                 'Religious Facilities (Synagogue, Church, Convent, Mosque)','Retirement / Nursing Homes','Ttc Admin Or Support Facility','Jails / Detention Centres',
                 'Pharmacy']
station=['Go Station','Ttc Light Rail Transit Station','Other Passenger Train Station','Ttc Subway Station']
trafficSector=['Other Passenger Train','Ttc Bus Stop / Shelter / Loop','Go Train','Other Train Admin Or Support Facility','Other Train Tracks',
               'Ttc Bus','Ttc Subway Train','Go Bus','Other Regional Transit System Vehicle']
nonCommercial = ['Other Non Commercial / Corporate Places (Non-Profit, Gov\'T, Firehall)', 'Police / Courts (Parole Board, Probation Office)']

other = ['Unknown']

new_df.loc[df_['Location_Type'].isin(outside), 
             'Premise_Type'] = 'Outside'
        
new_df.loc[df_['Location_Type'].isin(house), 
             'Premise_Type'] = 'House'

new_df.loc[df_['Location_Type'].isin(commercial), 
             'Premise_Type'] = 'Commercial'
        
new_df.loc[df_['Location_Type'].isin(apartment), 
             'Premise_Type'] = 'Apartment'
        
new_df.loc[df_['Location_Type'].isin(privateProperty), 
             'Premise_Type'] = 'PrivateProperty'        
        
new_df.loc[df_['Location_Type'].isin(station), 
             'Premise_Type'] = 'Station'
new_df.loc[df_['Location_Type'].isin(trafficSector), 
             'Premise_Type'] = 'TrafficSector'     
           
new_df.loc[df_['Location_Type'].isin(nonCommercial), 
             'Premise_Type'] = 'NonCommercial'       
new_df.loc[df_['Location_Type'].isin(other), 
             'Premise_Type'] = 'Other'       
           
#check data frame new_df after group

unique_col_values(new_df)           
print(new_df.head())

#create new data frame:
newCol=['Location_Type', 'Premise_Type','Bike_Type','Status']
df_fn=new_df[newCol]   

df_fn.Premise_Type.value_counts(dropna=False)
len(df_fn)

data_team2['Premise_Type'] = df_fn['Premise_Type']
data_team2.Premise_Type.value_counts(dropna=False)

"""================================= Bike_Type ==========================="""
data_bike = pd.DataFrame(data_team2, columns = ['Bike_Make', 'Bike_Model', 'Bike_Type']) 

#= Model ============

data_bike['Bike_Model'] = data_bike['Bike_Model'].str.replace(' ', '')

df = data_bike.groupby(['Bike_Model', 'Bike_Type'])['Bike_Type'].count().reset_index(name="Count")
print(df)

# High Frequency of Type for Model
dff = df[df.groupby('Bike_Model')['Count'].transform('max') == df['Count']]
print (dff)

# Put High Frequency of Type for that Model To "OT" Type
for i in data_bike.index:
    if data_bike['Bike_Type'][i] in ['OT', 'UN']:
        #print("i :", i, " === Count: ", dff[dff['Bike_Model'] == data_bike['Bike_Model'][i]]['Bike_Type'].count())
        if dff[dff['Bike_Model'] == data_bike['Bike_Model'][i]]['Bike_Type'].count() == 1:
            #print('Bike_Model: ', data_bike['Bike_Model'][i], 'Original Bike_Type: ', data_bike['Bike_Type'][i],
            #  'New Bike_Type: ', dff[dff['Bike_Model'] == data_bike['Bike_Model'][i]]['Bike_Type'].tolist()[0])
            data_bike['Bike_Type'][i] = dff[dff['Bike_Model'] == data_bike['Bike_Model'][i]]['Bike_Type'].tolist()[0]


data_bike['Bike_Type'].value_counts() #======================================= OT 2411 =========


#--- Bike ----------
data_bike['Bike_Make'] = data_bike['Bike_Make'].str.replace(' ', '')
data_bike['Bike_Type'].value_counts() #======================================= OT 2987 =========




df_make = data_bike.groupby(['Bike_Make', 'Bike_Type'])['Bike_Type'].count().reset_index(name="Count")
print(df_make)

# High Frequency of Type for Model
dff_make = df_make[df_make.groupby('Bike_Make')['Count'].transform('max') == df_make['Count']]
print (dff_make)



# Put High Frequency of Type for that Make To "OT" Type
for i in data_bike.index:
    if data_bike['Bike_Type'][i] in ['OT', 'UN']:
        #print("i :", i, " === Count: ", dff[dff['Bike_Model'] == data_bike['Bike_Model'][i]]['Bike_Type'].count())
        if dff_make[dff_make['Bike_Make'] == data_bike['Bike_Make'][i]]['Bike_Type'].count() == 1:
            #print('Bike_Model: ', data_bike['Bike_Model'][i], 'Original Bike_Type: ', data_bike['Bike_Type'][i],
            #  'New Bike_Type: ', dff[dff['Bike_Model'] == data_bike['Bike_Model'][i]]['Bike_Type'].tolist()[0])
            data_bike['Bike_Type'][i] = dff_make[dff_make['Bike_Make'] == data_bike['Bike_Make'][i]]['Bike_Type'].tolist()[0]


data_bike['Bike_Type'].value_counts() #======================================= OT 193 =========

data_team2['Bike_Type'] = data_bike['Bike_Type']
data_team2['Bike_Type'].value_counts(dropna=False)


"""======================================= Bike_Speed ================================"""
data_team2['Bike_Speed'].isnull().values.any()

"""======================================= Bike_Colour ================================"""
len(data_team2)
print('Bike_Colour', data_team2['Bike_Colour'].unique())

count = data_team2['Bike_Colour'].value_counts(dropna=False)

data_team2['Bike_Colour'] = data_team2['Bike_Colour'].str.replace(' ','')
data_team2['Bike_Colour'].fillna("BLK",inplace=True)

data_colour = []
for i in data_team2['Bike_Colour'].index:
    if data_team2['Bike_Colour'][i] not in ['BLK', 'BLU', 'GRY', 'WHI', 'RED']:
        data_colour.append('OTH')
    else:
        data_colour.append(data_team2['Bike_Colour'][i])


data_team2['Bike_Colour'] = data_colour
data_team2['Bike_Colour'].value_counts(dropna=False)


"""======================================= Cost_of_Bike ================================"""
data_team2.Cost_of_Bike.value_counts(dropna=False)
# NaN : 1311
data_team2.fillna(data_team2['Cost_of_Bike'].median(axis=0,skipna = True),inplace=True)
data_team2.Cost_of_Bike.value_counts(dropna=False)

"""======================================= Status ================================"""
#data_team2['Status'] = data_team2['Status'].replace(' ', '')

#data_team2['Status'] = data_team2['Status'].replace('STOLEN', 0)
#data_team2['Status'] = data_team2['Status'].replace('RECOVERED', 1)
#data_team2['Status'].value_counts(dropna=False)



"""========================== Updated City ===================="""

data_team2.loc[(data_team2['Hood_ID']<=20),'region'] = 'Etobicoke'
data_team2.loc[(data_team2['Hood_ID']>20) & (data_team2['Hood_ID']<=53),'region'] = 'Northyork'
data_team2.loc[(data_team2['Hood_ID']>53) & (data_team2['Hood_ID']<=61),'region'] = 'EastYork'
data_team2.loc[(data_team2['Hood_ID']>61) & (data_team2['Hood_ID']<=105),'region'] = 'OldToronto'
data_team2.loc[(data_team2['Hood_ID']>105) & (data_team2['Hood_ID']<=115),'region'] = 'York'
data_team2.loc[(data_team2['Hood_ID']>115) & (data_team2['Hood_ID']<=140),'region'] = 'Scarborough'


"""======================================= City ================================"""
# code is already run because it took 2 hours using geocode api
"""
from urllib.request import urlopen
import json
def getplace(lat, lon):
    url = "https://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    url += "&key=AIzaSyDvYr6M5y7clXeMY1bcZNP3t5uz5zeP-qM"
    v = urlopen(url).read()
    j = json.loads(v)
    components = j['results'][0]['address_components']
    components_2 = []
    city = None
    for c in components:
        if "sublocality_level_1" in c['types']:
            city = c['long_name']
        if city == None:
            #print("city is None");
            if "locality" in c['types']:
                city = c['long_name']
            else:
                components_2 = j['results'][1]['address_components']
                
    #print("components_2 : ", components_2, "\n\n")
    #print("components_2 len : ", len(components_2), "\n\n")
    
    if len(components_2) > 0:
        #print("components_2 is not null !")
        for c in components_2:
            if "locality" in c['types']:
                city = c['long_name']
                
        
    return city 

# Test
print(getplace(43.7872849,-79.4700851))

#data_latlong = data_team2[{'Lat', 'Long'}]
import pandas as pd
import os
path = "/Users/kyungjin/Downloads/Semester 5/COMP309 - Data Warehouse & Mining HCIS/Project 2"
filename = 'data_latlong_with_cities2.csv'
fullpath = os.path.join(path,filename)
data_latlongcity = pd.read_csv(fullpath)

print(data_latlongcity)

cities = []
for i in data_latlongcity.index:
    if pd.isnull(data_latlongcity['City'][i]):
        #print(i, " : ", data_latlongcity['City'][i])
        cities.append(getplace(data_latlongcity['Lat'][i], data_latlongcity['Long'][i]))
    else:
        cities.append(data_latlongcity['City'][i])
    
# 17892 rows

df_cities = pd.DataFrame(cities, columns =['City']) 
df_cities.to_csv (r'/Users/kyungjin/Downloads/Semester 5/COMP309 - Data Warehouse & Mining HCIS/Project 2/df_cities3.csv', index = None, header=True) 

data_latlongcity['City'] = df_cities

data_latlongcity.to_csv (r'/Users/kyungjin/Downloads/Semester 5/COMP309 - Data Warehouse & Mining HCIS/Project 2/data_latlong_with_cities3.csv', index = None, header=True) 
"""
filename = 'data_latlong_with_cities3.csv'
fullpath = os.path.join(path,filename)
data_cities = pd.read_csv(fullpath)

data_team2['City'] = data_cities['City']

"""============================ Status >> Drop “Unknown” rows !!!   ============================ """
len(data_team2) #17892
data_team2['Status'].value_counts() #Unknown 314
# should just have 17892 - 314 rows = 17578
data_team2 = data_team2[data_team2.Status != 'UNKNOWN']

len(data_team2) #17578
data_team2['Status'].value_counts() #Unknown 0


"""============================ Create the dummy variables ============================ """

cat_vars=['Season','Time','City','Premise_Type','Bike_Type']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(data_team2[var], prefix=var)
    data_team2_temp = data_team2.join(cat_list)
    data_team2 = data_team2_temp
data_team2.head(5)    

"""============================ Remove the original columns ============================ """
remove_vars = ['X', 'Y', 'Index_', 'event_unique_id', 'Primary_Offence', 'Occurrence_Date',
               'Occurrence_Year', 'Occurrence_Month', 'Occurrence_Day', 'Occurrence_Time',
               'Division', 'City', 'Location_Type', 'Premise_Type', 'Bike_Make', 
               'Bike_Model', 'Bike_Type', #'Bike_Speed', 
               'Bike_Colour', #'Cost_of_Bike',
               #'Status', 
               'Neighbourhood', 'Hood_ID', 'Lat', 'Long', 'ObjectId'
               ,'Season', 'Time']

data_team2_vars = data_team2.columns.values.tolist()
print(data_team2_vars)

to_keep=[i for i in data_team2_vars if i not in remove_vars]
print(to_keep)

data_team2_final = data_team2[to_keep]
data_team2_final.columns.values

"""====================================================================================
    Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
"""

# 3- Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
data_team2_final_vars = data_team2_final.columns.values.tolist()
print(data_team2_final_vars)
Y=['Status']
X=[i for i in data_team2_final_vars if i not in Y ]
print(X)
print(Y)

type(X)
type(Y)

print(data_team2_final.dtypes)
data_team2['Status'] = data_team2['Status'].replace(' ', '')
data_team2_final.loc[data_team2_final['Status'].str.contains('STOLEN'),'newStatus'] = 0
data_team2_final.loc[data_team2_final['Status'].str.contains('RECOVERED'),'newStatus'] = 1
print(data_team2_final.dtypes)

data_team2_final['Status'] = data_team2_final['newStatus']
print(data_team2_final.dtypes)

data_team2_final = data_team2_final.drop(['newStatus'], axis=1)


"""
Logistic Regression Model
    -	Carry out feature selection and update the data, as follows:
        a.	Carry out feature selection using the REF module from sklearn.model_selection 
            to select only 12 feature
        b.	Update X and Y to reflect only 12 features
"""
# carryout feature selection
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model)
rfe = rfe.fit(data_team2_final[X],data_team2_final[Y] )
#print(rfe.support_)
#print(rfe.ranking_)

rank = rfe.ranking_

length = len(X)
len(rank)

# Update X and Y with selected features
cols = []
for i in range(length):
    print(X[i], " rank is : ", rank[i])
    if rank[i] == 1:
        cols.append(X[i]) 

print(X)
print(cols)

X=data_team2_final[cols]
y=data_team2_final['Status']



from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import numpy as np
# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

##########################################################################################

clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(X_train,  y_train)
#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
#4-Check model accuracy
print (metrics.accuracy_score(y_test, predicted))	
metrics.f1_score(y_test, predicted, average='weighted', labels=np.unique(predicted))
metrics.precision_score(y_test, predicted, average='weighted')
metrics.recall_score(y_test, predicted, average='weighted')


"""============================== Resampling Techniques — Oversample minority class"""
from sklearn.utils import resample

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
stolen = X[X.Status==0]
recovered = X[X.Status==1]

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
y_train = upsampled.Status
X_train = upsampled.drop('Status', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)

# Checking accuracy
metrics.accuracy_score(y_test, upsampled_pred)
metrics.f1_score(y_test, upsampled_pred)
metrics.precision_score(y_test, upsampled_pred)
metrics.recall_score(y_test, upsampled_pred)


"""==============================4. Resampling techniques — Undersample majority class"""
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
y_train = downsampled.Status
X_train = downsampled.drop('Status', axis=1)

undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)

# Checking accuracy
metrics.accuracy_score(y_test, undersampled_pred)
metrics.f1_score(y_test, undersampled_pred)
metrics.precision_score(y_test, undersampled_pred)
metrics.recall_score(y_test, undersampled_pred)

"""============================== 5. Generate synthetic samples"""
from imblearn.over_sampling import SMOTE

# Separate input features and target
y = data_team2_final.Status
X = data_team2_final.drop('Status', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = smote.predict(X_test)

# Checking accuracy
metrics.accuracy_score(y_test, smote_pred)
metrics.f1_score(y_test, smote_pred)
metrics.precision_score(y_test, smote_pred)
metrics.recall_score(y_test, smote_pred)


"""============================== Decision Tree ======================"""
predictors = cols
target = ['Status']

import numpy as np
data_team2_final['is_train'] = np.random.uniform(0, 1, len(data_team2_final)) <= .75
print(data_team2_final.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_team2_final[data_team2_final['is_train']==True], data_team2_final[data_team2_final['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


"""
4-	Build the decision tree using the training dataset. 
    Use enotrpy as a method for splitting, and split only when reaching 20 matches.
"""
from sklearn.tree import DecisionTreeClassifier
dt_kyungjin = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kyungjin.fit(train[predictors], train[target])

"""
5-	Test the model using the testing dataset and 
    calculate a confusion matrix this time using pandas
"""
preds=dt_kyungjin.predict(test[predictors])
pd.crosstab(test['Status'],preds,rownames=['Actual'],colnames=['Predictions'])

"""
6-	Generate a dot file and visualize the tree using the online vizgraph 
    editor and share (download) as picture.
    
    https://dreampuf.github.io/GraphvizOnline
"""
from sklearn.tree import export_graphviz
with open('/Users/kyungjin/Downloads/Semester 5/COMP309 - Data Warehouse & Mining HCIS/Lab/Week 10 - Decision trees/Chapter 8/dtree3.dot', 'w') as dotfile:
    export_graphviz(dt_kyungjin, out_file = dotfile, feature_names = predictors)
dotfile.close()


print(test[predictors])
print(dt_kyungjin.predict(test[predictors]))














X=data_team2[predictors]
Y=data_team2[target]
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.25)


from sklearn.tree import DecisionTreeClassifier

for depth in range(1, 11):
  dt1_kyungjin = DecisionTreeClassifier(criterion='entropy',max_depth=depth, min_samples_split=20, random_state=99)
  dt1_kyungjin.fit(trainX,trainY)
  print("feature_importances_: ", dt1_kyungjin.feature_importances_)
# 10 fold cross validation using sklearn and all the data i.e validate the data 
  from sklearn.model_selection import KFold
#help(KFold)
  crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
  from sklearn.model_selection import cross_val_score
  score = np.mean(cross_val_score(dt1_kyungjin, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
  print("max_depth = ",depth,"score=",score)



### Test the model using the testing data
  testY_predict = dt1_kyungjin.predict(testX)
  testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
  
  print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
  print("F1 Score:",metrics.f1_score(testY, testY_predict))
  print("Precision_score:",metrics.precision_score(testY, testY_predict))
  print("Recall_score:",metrics.recall_score(testY, testY_predict))

