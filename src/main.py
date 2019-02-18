# load data
#Important: the pwd should be in src folder

# load packages
# load packages
import os
import time
import datetime
import pickle as p
import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from script import feature_selection_by_rfe, transformation, evaluation

#Important: the pwd should be in src folder
AQ_201804 = pd.read_csv('data/aiqQuality_201804.csv')
AQ_201701201801 = pd.read_csv('data/airQuality_201701-201801.csv')
AQ_20180203 = pd.read_csv('data/airQuality_201802-201803.csv')
GW_201804 = pd.read_csv('data/gridWeather_201804.csv')
GW_201701201803 = pd.read_csv('data/gridWeather_201701-201803.csv')
OW_201804 = pd.read_csv('data/observedWeather_201804.csv')
OW_201701201801 = pd.read_csv('data/observedWeather_201701-201801.csv')
OW_201802201803 = pd.read_csv('data/observedWeather_201802-201803.csv')
station_detail = pd.read_csv('data/station_detail.csv')

# join AQ data
AQ_201804.rename(index=str, columns={'station_id': 'stationId', 'time': 'utc_time',\
 'PM25_Concentration': 'PM2.5', 'PM10_Concentration': 'PM10', 'NO2_Concentration': 'NO2',\
 'CO_Concentration': 'CO', 'O3_Concentration':'O3', 'SO2_Concentration': 'SO2'}, inplace=True)
AQ_201804.drop(columns=['id'],inplace=True)
AQ = pd.concat([AQ_201701201801, AQ_20180203, AQ_201804], axis=0, join='outer', sort=False,ignore_index=True) 
AQ.rename(index=str, columns={'stationId': 'station_id'},inplace=True)
AQ = pd.merge(AQ, station_detail, how='left', on='station_id')
AQ['longitude'] = np.round(AQ['longitude'], 1)
AQ['latitude'] = np.round(AQ['latitude'], 1)
print('before dropping duplicate', AQ.shape)
AQ.drop_duplicates(['utc_time', 'station_id'], inplace=True)
print('after dropping duplicate', AQ.shape)
# AQ.info()
# AQ.head()

# join gird weather data
GW_201804.drop(columns=['id'],inplace=True)
GW_201701201803.rename(index=str, columns={'stationName': 'station_id', 'wind_speed/kph': 'wind_speed'}, inplace=True)
GW_longitude_latitude = GW_201701201803.loc[:,['station_id', 'longitude', 'latitude']].drop_duplicates()
# GW_longitude_latitude.info()
GW_201804.rename(index=str, columns={'time': 'utc_time'},inplace=True)
GW_201804 = GW_201804.merge(GW_longitude_latitude, on='station_id', how='left')
GW = pd.concat([GW_201701201803,GW_201804], axis=0, join='outer', sort=False,ignore_index=True)
GW_rename = GW.rename(columns={k: k+'_g' for k in GW.columns.tolist() if k not in ['station_id', 'utc_time']})
GW_rename.rename(index=str, columns={'station_id': 'grid_id'}, inplace=True)
print('before dropping duplicate', GW_rename.shape)
GW_rename.drop_duplicates(['utc_time', 'longitude_g', 'latitude_g'], inplace=True)
print('after dropping duplicate', GW_rename.shape) #(7498155, 10)

# GW_rename.info()
# GW_rename.head()

# join observed weather data
# OW_201701201801.info()
# OW_201802201803.info()
# OW_201804.info()
OW_201804.drop(columns=['id'],inplace=True)
OW_201804.rename(index=str, columns={'time': 'utc_time'},inplace=True)
OW = pd.concat([OW_201701201801, OW_201802201803, OW_201804], axis=0, join='outer', sort=False,ignore_index=True )
OW_rename = OW.rename(columns={k: k+'_o' for k in OW.columns.tolist() if k not in ['station_id', 'utc_time']})
OW_rename['latitude_o'] = np.round(OW_rename['latitude_o'], 1)
OW_rename['longitude_o'] = np.round(OW_rename['longitude_o'], 1)
OW_rename['latitude_o'] = np.round(OW_rename['latitude_o'], 1)
print('before dropping duplicate', OW_rename.shape)
OW_rename.drop_duplicates(['utc_time', 'longitude_o', 'latitude_o'], inplace=True)
OW_rename = OW_rename[(OW_rename['longitude_o'].notnull()) & (OW_rename['latitude_o'].notnull())]
print('after dropping duplicate', OW_rename.shape)

df_id_time = pd.DataFrame(columns=['station_id_l', 'utc_time_l'])
for station in AQ['station_id'].unique():
    df_slice = pd.DataFrame({'utc_time_l': pd.date_range(start=datetime.datetime(2017,1,1,0,0), end=datetime.datetime(2018,4,30,23,0), freq='H')})
    df_slice['station_id_l'] = station
    df_id_time = df_id_time.append(df_slice, sort=True)
df_id_time['utc_time_l'] = df_id_time['utc_time_l'].astype(str)
df_based = df_id_time.merge(AQ[['longitude', 'latitude','station_id']].drop_duplicates(), left_on=['station_id_l'], right_on=['station_id'], how='left')
df_based.drop('station_id', axis=1, inplace=True)
print(df_based.shape)

df_based_AQ = df_based.merge(AQ.drop(['longitude', 'latitude'], axis=1), left_on=['station_id_l', 'utc_time_l'], right_on=['station_id', 'utc_time'], how='left')
print('before dropping duplicate', df_based_AQ.shape)
df_based_AQ.drop_duplicates(['utc_time_l', 'station_id_l'], inplace=True)
df_based_AQ.drop(['station_id', 'utc_time'], axis=1, inplace=True)
df_based_AQ = df_based_AQ[(df_based_AQ['longitude'].notnull()) & (df_based_AQ['latitude'].notnull()) & (df_based_AQ['station_id_l'].notnull())]
print('after dropping duplicate', df_based_AQ.shape) #(377266, 13)

df_based_AQ_GW = df_based_AQ.merge(GW_rename, left_on=['longitude', 'latitude', 'utc_time_l'], right_on=['longitude_g', 'latitude_g', 'utc_time'], how='outer')
df_based_AQ_GW['longitude'] = df_based_AQ_GW['longitude'].fillna(df_based_AQ_GW['longitude_g'])
df_based_AQ_GW['latitude'] = df_based_AQ_GW['latitude'].fillna(df_based_AQ_GW['latitude_g'])
df_based_AQ_GW.drop(['longitude_g', 'latitude_g', 'grid_id', 'utc_time'], axis=1, inplace=True)
df_based_AQ_GW.rename(columns={'station_id_l': 'station_id', 'utc_time_l': 'utc_time'}, inplace=True)
print('before dropping duplicate', df_based_AQ_GW.shape)
df_based_AQ_GW.drop_duplicates(['utc_time', 'station_id'], inplace=True)
df_based_AQ_GW = df_based_AQ_GW[df_based_AQ_GW['station_id'].notnull()]
print('after dropping duplicate', df_based_AQ_GW.shape) #(407400, 18)
print(pd.isnull(df_based_AQ_GW).sum()/len(df_based_AQ_GW))

# df_based_AQ_GW.to_csv('data_integration/df_based_AQ_GW.csv', index_label=False)

#========================================================================================
#========================================================================================
#========================================================================================
#script to find out the other closest stations for each station

df_based_AQ_GW = pd.read_csv('data_integration/df_based_AQ_GW.csv')
df_station_id = df_based_AQ_GW[['station_id', 'longitude', 'latitude']].drop_duplicates()
df_station_id.index = df_station_id['station_id']

station_list = list(df_station_id['station_id'])
for station in station_list:
    df_station_id[station] = np.where(df_station_id['station_id']==station, np.nan, np.sqrt(np.square(df_station_id['longitude'] - df_station_id.loc[station, 'longitude']) + np.sqrt(np.square(df_station_id['latitude'] - df_station_id.loc[station, 'latitude']))))

for i in range(1, 7): #6 cloestest stations
    for station in station_list:
        df_station_id.loc[station, 'closest_station'+str(i)] = df_station_id[station].sort_values().index[i]

station_id_closest = df_station_id[[col for col in df_station_id if 'closest_station' in col]]
station_id_closest.to_csv('data_integration/station_id_closest.csv')

#========================================================================================
#========================================================================================
#========================================================================================
#For building the models, please refer the 3 scripts
def predict(df_april_last_hours, May_1_2, model_dict, cto_dict, imp_dict, scalez_dict, rfe_dict, closest_station_number=6, random_state=4):
    """
    predict the 3 pollutions for for 1st and 2nd May.

    Parameters
    ----------
    df_april_last_hours : pandas.DataFrame
        input data
    May_1_2 : pandas.DataFrame
        The input data of 1st and 2nd May 2018
    model_dict : dict
        the model dictionary of each target
    rfe_dict : dict
        the rfe.pickle dictionary of each target
    closest_station_number : int
        the number of closest station in feature engineering
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    pandas.DataFrame
        the data with predicted 3 targets
    """
    final_data = pd.DataFrame()

    for station_id in df_april_last_hours['station_id'].unique():
    # for station_id in ['dongsihuan_aq']:
        subcase = df_april_last_hours[df_april_last_hours['station_id']==station_id]
        May_1_2_subcase = May_1_2[May_1_2['station_id']==station_id] #48 rows
        feature_data = pd.concat([subcase, May_1_2_subcase], sort=True).sort_values('utc_time') #48+8 rows
        number_of_record_before = len(subcase)
        feature_data.index = range(-len(subcase), len(May_1_2_subcase)) #index = 0 for 2018-05-01 00:00 #len(subcase) is the number of hour before 2018-05-01 00:00
        for i in range(len(May_1_2_subcase)): #48 hours
            for target in ['O3', 'PM10', 'PM2.5']: #3 targets
                feature_data_after = transformation.feature_engineering(df=feature_data.loc[i-number_of_record_before:i, ], target=target, closest_station_number=closest_station_number, drop=False)
                feature_data_after.reindex(list(feature_data_after.index))
                feature_data_after2 = feature_data_after.drop([t for t in ['PM2.5', 'PM10', 'O3'] if t != target], axis=1)
                feature_data_after3 = cto_dict[target].transform(feature_data_after2)
                feature_data_after4 = pd.DataFrame(imp_dict[target].transform(feature_data_after3), index=feature_data_after3.index, columns=feature_data_after3.columns)
                feature_data_after5 = feature_data_after4.drop(target, axis=1)
                for col in feature_data_after5:
                    feature_data_after5[col] = feature_data_after5[col].astype(float)
                feature_data_after6 = pd.DataFrame(data=scalez_dict[target].transform(feature_data_after5), index=feature_data_after5.index, columns=feature_data_after5.columns)
                feature_table_rfe = rfe_dict[target]
                feature_data_after7 = feature_data_after6[feature_table_rfe[0][feature_table_rfe[0]['XGBRegressor_rank'] <= feature_table_rfe[1]['final_number']].index.tolist()]
                feature_data.loc[i, target] = model_dict[target].predict(pd.DataFrame(feature_data_after7.loc[i, :]).T)[0]
        final_data = final_data.append(feature_data[feature_data.index >= 0])
    return final_data

target_list = ['O3', 'PM10', 'PM2.5']
outlier_threshold = '1'
other_threshold = 30
model = 'XGBRegressor'

model_dict = {}
cto_dict = {}
imp_dict = {}
rfe_dict = {}
scalez_dict = {}
for target, random_state, model_random_state, X_column_length in zip(target_list, [7, 7, 7], [7, 7, 7], [82, 82, 82]):
    with open(os.path.join('model', '_'.join([target, str(model).split('(')[0], 'seed',  str(random_state), 'seed_for_model',  str(model_random_state), 'outlier', outlier_threshold+'_.pickle'])), "rb") as file:
        model_dict[target] = p.load(file)
    with open('input/imputer_'+target+'_seed_'+str(random_state)+'.pickle', "rb") as file:
        imp_dict[target] = p.load(file)
    with open('input/categorytoordinal_'+target+'_'+str(other_threshold)+'_seed_'+str(random_state)+'.pickle', "rb") as file:
        cto_dict[target] = p.load(file)
    with open('input/standardscaler_'+target+'_'+str(other_threshold)+'_seed_'+str(random_state)+'.pickle', "rb") as file:
        scalez_dict[target] = p.load(file)
    with open(os.path.join('evaluation', target, target+'_'+str(X_column_length)+'_len_table'+'_seed_'+str(random_state)+'_outlier_'+outlier_threshold+'.pickle'), 'rb') as file:
        rfe_dict[target] = p.load(file)

df_based_AQ_GW = pd.read_csv('data_integration/df_based_AQ_GW.csv')
df_raw = df_based_AQ_GW.copy()
df = df_raw.drop(['CO', 'SO2', 'NO2'], axis=1)
df.drop_duplicates(['utc_time', 'station_id'], inplace=True)

May_1_2 = pd.read_csv('data_integration/May_1_2.csv')
rename_dict = {col: col+'_g' for col in ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather']}
rename_dict['time'] = 'utc_time'
May_1_2.rename(columns=rename_dict, inplace=True)
May_1_2.drop(['grid_id', 'longitude', 'latitude'], axis=1, inplace=True)

#extract the last day (30th April) for input data, since we request to have previous timestamp
df_april_last_hours = df[df['utc_time'] >= str(datetime.datetime(2018, 4, 30, 0, 0))]
for col in ['PM2.5', 'PM10', 'O3']:
    df_april_last_hours.loc[:, col] = df.groupby(['utc_time'], sort=False)[col].transform(lambda x: x.fillna(x.mean()))

print(df_april_last_hours.shape) #(840, 14)

submission_df = predict(df_april_last_hours, May_1_2, model_dict, cto_dict, imp_dict, scalez_dict, rfe_dict, closest_station_number=6, random_state=4)

submission_df2 = submission_df.copy()
submission_df2.loc[:, 'utc_time'] = pd.to_datetime(submission_df2['utc_time'], errors='coerce')
submission_df2['time_id'] = (submission_df2['utc_time'].dt.day - 1)*24 + submission_df2['utc_time'].dt.hour
submission_df2['test_id'] =  submission_df2['station_id'] + '#' + submission_df2['time_id'].astype(str)
submission_df3 = submission_df2[['test_id', 'PM2.5', 'PM10', 'O3']]
submission_df3.to_csv('data_integration/submission_xgb.csv', index=False)

submission_xgb = pd.read_csv('data_integration/submission_xgb.csv')
submission_xgb.drop('O3', axis=1, inplace=True)
#We have found that O3 is very hard to predict and have developed another strategies features in prediction_lightgbm.py
submission_lgb = pd.read_csv('data_integration/submission_lgb.csv')
submission_lgb = submission_lgb[['test_id', 'O3']]
submission = pd.merge(submission_xgb, submission_lgb, left_on=['test_id'], right_on=['test_id'])
submission.to_csv(os.path.join(os.pardir, 'submission.csv'), index=False)
