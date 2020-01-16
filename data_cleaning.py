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

path = "/Users/kyungjin/find_bicycle"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_bycycle = pd.read_csv(fullpath)

"""=========================== Occurrence_Month ==========================="""
data_bycycle.loc[(data_bycycle['Occurrence_Month'] < 3),'Season'] = 'Winter'
data_bycycle.loc[(data_bycycle['Occurrence_Month'] == 12),'Season'] = 'Winter'
data_bycycle.loc[(data_bycycle['Occurrence_Month'] >= 3) & (data_bycycle['Occurrence_Month'] < 6),'Season'] = 'Spring'
data_bycycle.loc[(data_bycycle['Occurrence_Month'] >= 6) & (data_bycycle['Occurrence_Month'] < 9),'Season'] = 'Summer'
data_bycycle.loc[(data_bycycle['Occurrence_Month'] >= 9) & (data_bycycle['Occurrence_Month'] < 12),'Season'] = 'Fall'

data_bycycle.Season.value_counts(dropna=False)

"""=========================== Occurrence_Time ==========================="""
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('00:'),'Time'] = 'Night'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('01:'),'Time'] = 'Night'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('02:'),'Time'] = 'Night'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('03:'),'Time'] = 'Night'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('04:'),'Time'] = 'Night'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('05:'),'Time'] = 'Night'

data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('06:'),'Time'] = 'Morning'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('07:'),'Time'] = 'Morning'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('08:'),'Time'] = 'Morning'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('09:'),'Time'] = 'Morning'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('10:'),'Time'] = 'Morning'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('11:'),'Time'] = 'Morning'

data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('12:'),'Time'] = 'Afternoon'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('13:'),'Time'] = 'Afternoon'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('14:'),'Time'] = 'Afternoon'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('15:'),'Time'] = 'Afternoon'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('16:'),'Time'] = 'Afternoon'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('17:'),'Time'] = 'Afternoon'

data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('18:'),'Time'] = 'Evening'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('19:'),'Time'] = 'Evening'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('20:'),'Time'] = 'Evening'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('21:'),'Time'] = 'Evening'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('22:'),'Time'] = 'Evening'
data_bycycle.loc[data_bycycle['Occurrence_Time'].str.contains('23:'),'Time'] = 'Evening'


data_bycycle['Time'].value_counts(dropna=False)

"""=========================== Premise_Type ==========================="""

categories = []
for col, col_type in data_bycycle.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          data_bycycle[col].fillna(0, inplace=True)

#Create new data set: 
include = ['Occurrence_Day', 'Occurrence_Month',
           'Bike_Type', 'Location_Type','Bike_Colour','Status']
df_ = data_bycycle[include]
df_ = data_bycycle[include]

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

data_bycycle['Premise_Type'] = df_fn['Premise_Type']
data_bycycle.Premise_Type.value_counts(dropna=False)

"""================================= Bike_Type ==========================="""
data_bike = pd.DataFrame(data_bycycle, columns = ['Bike_Make', 'Bike_Model', 'Bike_Type']) 

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

data_bycycle['Bike_Type'] = data_bike['Bike_Type']
data_bycycle['Bike_Type'].value_counts(dropna=False)


"""======================================= Bike_Speed ================================"""
data_bycycle['Bike_Speed'].isnull().values.any()

"""======================================= Bike_Colour ================================"""
len(data_bycycle)
print('Bike_Colour', data_bycycle['Bike_Colour'].unique())

count = data_bycycle['Bike_Colour'].value_counts(dropna=False)

data_bycycle['Bike_Colour'] = data_bycycle['Bike_Colour'].str.replace(' ','')
data_bycycle['Bike_Colour'].fillna("BLK",inplace=True)

data_colour = []
for i in data_bycycle['Bike_Colour'].index:
    if data_bycycle['Bike_Colour'][i] not in ['BLK', 'BLU', 'GRY', 'WHI', 'RED']:
        data_colour.append('OTH')
    else:
        data_colour.append(data_bycycle['Bike_Colour'][i])


data_bycycle['Bike_Colour'] = data_colour
data_bycycle['Bike_Colour'].value_counts(dropna=False)


"""======================================= Cost_of_Bike ================================"""
data_bycycle.Cost_of_Bike.value_counts(dropna=False)
# NaN : 1311
data_bycycle.Cost_of_Bike.fillna(data_bycycle['Cost_of_Bike'].median(axis=0,skipna = True),inplace=True)
data_bycycle.Cost_of_Bike.value_counts(dropna=False)

"""======================================= City ================================"""
data_bycycle.loc[(data_bycycle['Hood_ID']<=20),'City'] = 'Etobicoke'
data_bycycle.loc[(data_bycycle['Hood_ID']>20) & (data_bycycle['Hood_ID']<=53),'City'] = 'NorthYork'
data_bycycle.loc[(data_bycycle['Hood_ID']>53) & (data_bycycle['Hood_ID']<=61),'City'] = 'EastYork'
data_bycycle.loc[(data_bycycle['Hood_ID']>61) & (data_bycycle['Hood_ID']<=105),'City'] = 'OldToronto'
data_bycycle.loc[(data_bycycle['Hood_ID']>105) & (data_bycycle['Hood_ID']<=115),'City'] = 'York'
data_bycycle.loc[(data_bycycle['Hood_ID']>115) & (data_bycycle['Hood_ID']<=140),'City'] = 'Scarborough'

"""============================ Status >> Drop “Unknown” rows !!!   ============================ """
len(data_bycycle) #17892
data_bycycle['Status'].value_counts() #Unknown 314
# should just have 17892 - 314 rows = 17578
data_bycycle = data_bycycle[data_bycycle.Status != 'UNKNOWN']

len(data_bycycle) #17578
data_bycycle['Status'].value_counts() #Unknown 0

"""============================ Cost >> Drop outlier rows !!!   ============================ """

data_bycycle = data_bycycle[data_bycycle.Cost_of_Bike != 19097]
data_bycycle = data_bycycle[data_bycycle.Cost_of_Bike != 23000]
data_bycycle = data_bycycle[data_bycycle.Cost_of_Bike != 28000]
data_bycycle = data_bycycle[data_bycycle.Cost_of_Bike != 30000]
data_bycycle = data_bycycle[data_bycycle.Cost_of_Bike != 42999]
data_bycycle = data_bycycle[data_bycycle.Cost_of_Bike != 90021]

"""============================ Create the dummy variables ============================ """

#cat_vars=['Season','Time','City','Premise_Type','Bike_Type','Bike_Colour']
cat_vars=['Season','Time','City','Premise_Type','Bike_Type']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(data_bycycle[var], prefix=var)
    data_bycycle_temp = data_bycycle.join(cat_list)
    data_bycycle = data_bycycle_temp
data_bycycle.head(5)    

"""============================ Remove the original columns ============================ """
remove_vars = ['X', 'Y', 'Index_', 'event_unique_id', 'Primary_Offence', 'Occurrence_Date',
               'Occurrence_Year', 'Occurrence_Month', 'Occurrence_Day', 'Occurrence_Time',
               'Division', 'City', 'Location_Type', 'Premise_Type', 'Bike_Make', 
               'Bike_Model', 'Bike_Type', #'Bike_Speed', 
               'Bike_Colour', #'Cost_of_Bike',
               #'Status', 
               'Neighbourhood', 'Hood_ID', 'Lat', 'Long', 'ObjectId'
               ,'Season', 'Time']

data_bycycle_vars = data_bycycle.columns.values.tolist()
print(data_bycycle_vars)

to_keep=[i for i in data_bycycle_vars if i not in remove_vars]
print(to_keep)

data_bycycle_final = data_bycycle[to_keep]
data_bycycle_final.columns.values

data_bycycle_final.dtypes

"""========================= Replace Status (Stolen 0, Recovered 1 ========================="""
print(data_bycycle_final.dtypes)
data_bycycle['Status'] = data_bycycle['Status'].replace(' ', '')
data_bycycle_final.loc[data_bycycle_final['Status'].str.contains('STOLEN'),'newStatus'] = 0
data_bycycle_final.loc[data_bycycle_final['Status'].str.contains('RECOVERED'),'newStatus'] = 1
print(data_bycycle_final.dtypes)

data_bycycle_final['Status'] = data_bycycle_final['newStatus']
print(data_bycycle_final.dtypes)

data_bycycle_final = data_bycycle_final.drop(['newStatus'], axis=1)