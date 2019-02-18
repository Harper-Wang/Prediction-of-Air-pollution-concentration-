#!/usr/bin/env python
# coding: utf-8

# # load packages
import matplotlib
matplotlib.use('TkAgg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import copy
import datetime
import pickle as p
import numpy as np 
import pandas as pd
pd.options.display.max_columns = 200
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
import lightgbm as lgb
from script import feature_selection_by_rfe, transformation, evaluation

from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
#%%


#  load data and initialize variables

# In[132]:
traindata = pd.read_csv('data_integration/train.csv')
traindata = traindata.drop(['CO', 'SO2', 'NO2'], axis=1)
traindata.loc[:, 'utc_time'] = pd.to_datetime(traindata['utc_time'], errors='coerce',dayfirst=True) 
traindata.drop_duplicates(['utc_time', 'station_id'], inplace=True)
traindata.info()
#%%
april2018 = traindata[traindata['utc_time'] >= str(datetime.datetime(2018, 4, 1, 0, 0))]
testdata = pd.read_csv('data_integration/test.csv')
testdata.loc[:, 'utc_time'] = pd.to_datetime(testdata['utc_time'], errors='coerce',dayfirst=True) 
targets = ['PM2.5', 'PM10', 'O3']


# # Feature Engineering

# In[111]:


def shifting(df, shift_col, groupby_col=None, shift_number=1):   
    if groupby_col is None:
        df.loc[:, shift_col+'(t-'+str(shift_number)+')'] = df[shift_col].shift(shift_number)
    else:
        df.loc[:, shift_col+'(t-'+str(shift_number)+')'] = df.groupby(groupby_col)[shift_col].shift(shift_number)
    return df

def transform1(df, target=None):
    df = df.copy()
    if target != None:
        col_todrop = [col for col in targets if col != target]
        df.drop(columns=col_todrop, inplace=True)
    else:
        df = df.copy()
    df.loc[:, 'month'] = df['utc_time'].dt.month
    # df.loc[:, 'day'] = df['utc_time'].dt.day
    df.loc[:, 'weekday'] = df['utc_time'].dt.weekday
    df.loc[:, 'hour'] = df['utc_time'].dt.hour
    
    numeric_columns = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
    categorical_columns = ['weather','station_class']
    groupby_col='station_id'
    #na treatment for numeric columns
    for col in df:
        if any(num_col in col for num_col in numeric_columns):
            if col != target:
                df.loc[:, col] = df.groupby(groupby_col)[col].apply(lambda group: group.interpolate())

    #na treatment for categorical columns
    for col in df:
        if any(cate_col in col for cate_col in categorical_columns):
            df.loc[:, col] = df.groupby(groupby_col)[col].fillna(method='ffill')
    
    for t in range(1, 5):
        for col in ['humidity', 'wind_speed', 'temperature', 'pressure', 'wind_direction']:
            if col in df:
                df = shifting(df, shift_col=col, groupby_col='station_id', shift_number=t)
    
    # add closest location
    df_station_id = pd.read_csv('data/station_id_closest.csv')
    df_station_id.rename(columns={'station_id': 'station_id_r'}, inplace=True)
    df = df.reset_index().merge(df_station_id,  how='left', left_on='station_id', right_on='station_id_r').drop('station_id_r', axis=1).set_index('index')

    df_closest = df[[col for col in ['station_id', 'utc_time']+numeric_columns if col in df]]
    for i in range(1, 3):
        rename_dict = {l: l+'_closest'+str(i) for l in numeric_columns}
        rename_dict['station_id'] = 'station_id_r'
        rename_dict['utc_time'] = 'utc_time_r'
        df_closest_reanme = df_closest.rename(columns=rename_dict)
        df = df.reset_index().merge(df_closest_reanme, how='left', left_on=['closest_station'+str(i), 'utc_time'], right_on=['station_id_r', 'utc_time_r']).drop(['station_id_r', 'utc_time_r'], axis=1).set_index('index')
    df.drop([col for col in df if 'closest_station' in col], axis=1, inplace=True) #drop the col containing the 'closest_station' wording

#     df.dropna(inplace=True)
    df.drop(columns=['station_id','utc_time'],inplace=True)
    print(df.shape)
    return df





#%%
def transform_traindata(df,target):
    # convert categorical data to numbers
    cto = categorytoordinal.CategoryToOrdinal(other_threshold=30)
    categorical_col = ['weather', 'station_class'] 
    cto.fit(df[categorical_col], df[target])
    df_transformed = cto.transform(df)
    with open('input/categorytoordinal_'+target+'.pickle', "wb") as f_out:
        p.dump(cto, f_out)
        
    df_X = df_transformed.drop(columns=[target])
    df_y = df_transformed[target]
    
    # normalize
    sc = StandardScaler()
    df_scalez = pd.DataFrame(data=sc.fit_transform(df_X), index=df_X.index, columns=df_X.columns)
    with open('input/standardscaler_'+target+'.pickle', "wb") as f_out:
        p.dump(sc, f_out)
    print('shape of scalez:', df_scalez.shape)

    return df_scalez, df_y

def transform_testdata(df, target):
    df['station_class'].fillna('urban',inplace=True)
    df['weather'].fillna('CLOUDY',inplace=True)
    with open('input/categorytoordinal_'+target+'.pickle', "rb") as file:
        cto = p.load(file)
    df = cto.transform(df)
    
    df.fillna(method='pad',inplace=True)
    
    with open('input/standardscaler_'+target+'.pickle', "rb") as file:
        scalez = p.load(file)

    df_X = df.drop(columns=[target])
    df_y = df[target]

    sc = StandardScaler()
    df_scalez = pd.DataFrame(data=sc.fit_transform(df_X), index=df_X.index, columns=df_X.columns)

    print('shape of scalez:', df_scalez.shape)

    return df_scalez


# In[112]:

# get training data
train_PM25 = transform1(april2018,'PM2.5')
train_PM25.dropna(inplace=True)
train_X_PM25, train_y_PM25 = transform_traindata(train_PM25,'PM2.5')

train_PM10 = transform1(april2018,'PM10')
train_PM10.dropna(inplace=True)
train_X_PM10, train_y_PM10 = transform_traindata(train_PM10,'PM10')

train_O3 = transform1(april2018,'O3')
train_O3.dropna(inplace=True)
train_X_O3, train_y_O3 = transform_traindata(train_O3,'O3')


# In[113]:


train_X_PM25.head()


# # Model Development 
# 1. PM2.5

# In[114]:


model_random_state = 5
params = {
    'max_depth': range(6,15),
    'num_leaves': range(20, 100, 10),
#     'min_child_samples': [18, 19, 20, 21, 22],
#     'min_child_weight':[0.001, 0.002],
    'feature_fraction': [0.5, 0.6, 0.7,0.8],
    'bagging_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'reg_alpha': [ 0.6, 0.7, 0.8, 0.9, 1],
    'reg_lambda': [ 0.6, 0.7, 0.8, 0.9, 1]
    }

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=40, max_depth=6,
                              metric='rmse', bagging_fraction = 0.5,feature_fraction = 0.4)
rc = RandomizedSearchCV(model_lgb, params, cv=5, random_state=model_random_state)
rc.fit(train_X_PM25,train_y_PM25)

#%%
print('best params: ', rc.best_params_)
print('best score: ', rc.best_score_)
best_position = rc.best_index_
print('best train score:', rc.cv_results_['mean_train_score'][best_position])
print('best train std:', rc.cv_results_['std_train_score'][best_position])
print('best test score:', rc.cv_results_['mean_test_score'][best_position])
print('best test std:', rc.cv_results_['std_test_score'][best_position])
# best params:  {'reg_lambda': 0.9, 'reg_alpha': 0.6, 'num_leaves': 60, 'max_depth': 14, 'feature_fraction': 0.5, 'bagging_fraction': 0.3}
# best score:  0.8011678911600456
# best train score: 0.9083334290707444
# best train std: 0.004247777959102244
# best test score: 0.8011678911600456
# best test std: 0.05801191776853861

# In[115]:


lgbr = lgb.LGBMRegressor(objective='regression')
updated_dict = rc.best_params_
updated_dict['learning_rate'] = .01
updated_dict['n_estimators'] = 400
updated_dict['reg_alpha'] = 1
updated_dict['reg_lambda'] = 1
lgbr.set_params(**updated_dict)
scores = evaluation.cv_scores(lgbr, train_X_PM25, train_y_PM25, cv=KFold(5, random_state=model_random_state), return_estimator=True)
final_model_PM25 = scores['estimator'][0]
# train mean of r2: 0.9105595243830468
# train std of r2: 0.004123702698672325
# test mean of r2: 0.8026625934690428
# test std of r2: 0.056931438217053924
# train mean of smape: 0.29400950618095223
# train std of smape: 0.007571337452039795
# test mean of smape: 0.3703110126085242
# test std of smape: 0.05225316940873302

#  2. PM10

# In[116]:


model_random_state = 5
params = {
    'max_depth': range(5,16),
    'num_leaves': range(20, 200, 10),
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002],
    'feature_fraction': [0.5, 0.6, 0.7,0.8],
    'bagging_fraction': [0.1, 0.2,0.3],
    'reg_alpha': [ 0,0.1,0.2,0.3, 0.4,0.5],
    'reg_lambda': [ 0,0.1,0.2,0.3,0.4, 0.5]
    }

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=43, max_depth=6,
                              metric='rmse', bagging_fraction = 0.5,feature_fraction = 0.4)
rc = RandomizedSearchCV(model_lgb, params, cv=5, random_state=model_random_state)
rc.fit(train_X_PM10,train_y_PM10)

#%%
print('best params: ', rc.best_params_)
print('best score: ', rc.best_score_)
best_position = rc.best_index_
print('best train score:', rc.cv_results_['mean_train_score'][best_position])
print('best train std:', rc.cv_results_['std_train_score'][best_position])
print('best test score:', rc.cv_results_['mean_test_score'][best_position])
print('best test std:', rc.cv_results_['std_test_score'][best_position])
# best params:  {'reg_lambda': 0.2, 'reg_alpha': 0.4, 'num_leaves': 150, 'min_child_weight': 0.002, 'min_child_samples': 21, 'max_depth': 11, 'feature_fraction': 0.7, 'bagging_fraction': 0.3}
# best score:  0.6650096420036282
# best train score: 0.8597124952930624
# best train std: 0.010064254262147538
# best test score: 0.6650096420036282
# best test std: 0.08855093716362965

# In[117]:


lgbr = lgb.LGBMRegressor(objective='regression')
updated_dict = rc.best_params_
updated_dict['learning_rate'] = .01
updated_dict['n_estimators'] = 400
updated_dict['reg_alpha'] = 1
updated_dict['reg_lambda'] = 1
updated_dict['feature_fraction'] = 0.7
updated_dict['bagging_fraction'] = 0.2
lgbr.set_params(**updated_dict)
scores = evaluation.cv_scores(lgbr, train_X_PM10, train_y_PM10, cv=KFold(3, random_state=model_random_state), return_estimator=True)
final_model_PM10 = scores['estimator'][0]
# train mean of r2: 0.8561773297780918
# train std of r2: 0.0033759152356996926
# test mean of r2: 0.658140903774905
# test std of r2: 0.022160144611612042
# train mean of smape: 0.18961377056288706
# train std of smape: 0.0050006602340263865
# test mean of smape: 0.2848443726309326
# test std of smape: 0.02199778022164099

#  3. O3

# In[118]:


model_random_state = 5
params = {
    'max_depth': range(5,16),
    'num_leaves': range(20, 200, 10),
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002],
    'feature_fraction': [0.5, 0.6, 0.7,0.8],
    'bagging_fraction': [0.1, 0.2,0.3,0.4,0.5],
    'reg_alpha': [ 0,0.1,0.2,0.3, 0.4,0.5],
    'reg_lambda': [ 0,0.1,0.2,0.3,0.4, 0.5]
    }

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=43, max_depth=6,
                              metric='rmse', bagging_fraction = 0.5,feature_fraction = 0.4)
rc = RandomizedSearchCV(model_lgb, params, cv=5, random_state=model_random_state)
rc.fit(train_X_O3,train_y_O3)

#%%
print('best params: ', rc.best_params_)
print('best score: ', rc.best_score_)
best_position = rc.best_index_
print('best train score:', rc.cv_results_['mean_train_score'][best_position])
print('best train std:', rc.cv_results_['std_train_score'][best_position])
print('best test score:', rc.cv_results_['mean_test_score'][best_position])
print('best test std:', rc.cv_results_['std_test_score'][best_position])
# best params:  {'reg_lambda': 0.1, 'reg_alpha': 0.2, 'num_leaves': 100, 'min_child_weight': 0.001, 'min_child_samples': 18, 'max_depth': 10, 'feature_fraction': 0.5, 'bagging_fraction': 0.5}
# best score:  0.8265967721167117
# best train score: 0.9318160687012567
# best train std: 0.0019137799948657874
# best test score: 0.8265967721167117
# best test std: 0.05937173848175587

# In[119]:


lgbr = lgb.LGBMRegressor(objective='regression')
updated_dict = rc.best_params_
updated_dict['learning_rate'] = .01
updated_dict['n_estimators'] = 500
updated_dict['reg_alpha'] = 1
updated_dict['reg_lambda'] = 1
updated_dict['feature_fraction'] = 0.7
updated_dict['bagging_fraction'] = 0.2
updated_dict['num_leaves'] = 80
lgbr.set_params(**updated_dict)
scores = evaluation.cv_scores(lgbr, train_X_O3, train_y_O3, cv=KFold(3, random_state=model_random_state), return_estimator=True)
final_model_O3 = scores['estimator'][0]
# train mean of r2: 0.9396256582701975
# train std of r2: 0.001976816243115814
# test mean of r2: 0.8150996742312225
# test std of r2: 0.02999523207368587
# train mean of smape: 0.2543138807417051
# train std of smape: 0.007211682809784926
# test mean of smape: 0.354179470156675
# test std of smape: 0.006811125626554038

# In[120]:


model_dict = {}
model_dict['PM2.5'] = final_model_PM25
model_dict['PM10'] = final_model_PM10
model_dict['O3'] = final_model_O3


# In[121]:


def predict(df_april_last_hours, May_1_2, model_dict, random_state=4):

    final_data = pd.DataFrame()
    for station_id in df_april_last_hours['station_id'].unique():
        subcase = df_april_last_hours[df_april_last_hours['station_id']==station_id]
        May_1_2_subcase = May_1_2[May_1_2['station_id']==station_id] #48 rows
        feature_data = pd.concat([subcase, May_1_2_subcase], sort=True).sort_values('utc_time') #48+8 rows
        number_of_record_before = len(subcase)
        feature_data.index = range(-len(subcase), len(May_1_2_subcase)) #index = 0 for 2018-05-01 00:00 #len(subcase) is the number of hour before 2018-05-01 00:00
        for target in targets:
            feature_data2 = transform1(feature_data,target)
            feature_data3 = transform_testdata(feature_data2,target)
            feature_data[target][24:72] = model_dict[target].predict(feature_data3.iloc[24:72, :])
#             print(feature_data3.iloc[24:72, :])
        final_data = final_data.append(feature_data[feature_data.index >= 0])
    return final_data


# In[122]:


# testing on april 29-30
df = april2018
test_last_2_days = df[df['utc_time'] >= str(datetime.datetime(2018,4, 29, 0, 0))]
april_28_last_hours = df[(df['utc_time'] < str(datetime.datetime(2018, 4, 29, 0, 0))) & (df['utc_time'] >= str(datetime.datetime(2018,4, 28, 0, 0)))]
for col in targets:
    april_28_last_hours.loc[:, col] = df.groupby(['utc_time'], sort=False)[col].transform(lambda x: x.fillna(x.mean()))

print(test_last_2_days.shape)
print(april_28_last_hours.shape)


# In[123]:


test_df = predict(april_28_last_hours, test_last_2_days, model_dict , random_state=7)


# In[124]:


targets_df = test_df[targets+['station_id', 'utc_time']]
targets_df.rename(columns={target: target+'_p' for target in targets}, inplace=True)
targets_df2 = pd.merge(df[targets+['station_id', 'utc_time']], targets_df, left_on=['station_id', 'utc_time'], right_on=['station_id', 'utc_time'])
print(targets_df2.shape)
for target in targets:
    targets_notnull = targets_df2[targets_df2[target].notnull()]
    print('smape of', target, ':', evaluation.smape(targets_notnull[target], targets_notnull[target+'_p']))
# (1680, 8)
# smape of PM2.5 : 0.6943130646881066
# smape of PM10 : 0.5578754615225703
# smape of O3 : 0.38913245397481866

# In[133]:


# prediction on 1-2 May
df = april2018
testdata['PM2.5']=[0 for i in range(testdata.shape[0])]
testdata['PM10']=[0 for i in range(testdata.shape[0])]
testdata['O3']=[0 for i in range(testdata.shape[0])]
april_30_last_hours = df[df['utc_time'] >= str(datetime.datetime(2018,4, 30, 0, 0))]
for col in targets:
    april_30_last_hours.loc[:, col] = df.groupby(['utc_time'], sort=False)[col].transform(lambda x: x.fillna(x.mean()))

print(testdata.shape)
print(april_30_last_hours.shape)


# In[134]:


prediction_df = predict(april_30_last_hours, testdata, model_dict , random_state=7)


# In[140]:


prediction_df.head()


# In[168]:


df = prediction_df[['station_id','utc_time','PM2.5','PM10','O3']]
df.index = range(df.shape[0])
df['hour'] = df['utc_time'].dt.hour
a=[]
for i in range(35):
    for j in range(48):
        a.append(j)
df['test_id'] = [df.loc[i,'station_id']+'#'+str(a[i]) for i in range(df.shape[0])] 
submission_lgb = df[['test_id','PM2.5','PM10','O3']]
submission_lgb.to_csv('submission_lgb.csv', index=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




