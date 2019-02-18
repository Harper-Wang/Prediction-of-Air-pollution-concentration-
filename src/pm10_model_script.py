#%%
"""
Predict concentration level of pm10.

Step1: Data Cleansing and Feature Engineering. 
 Use functions defined in 'transformation.py' to transform the raw data. Details can be found in 'transformation.py'

Step2：Data Resampling
Resample some data from the whole dataset to get fewer input samples. Note that we used sample weights to pick samples 
     
Step3: Feature Selection.
Feature ranking with recursive feature elimination(RFE) using sample data from Step2.

Step3：Data Resampling
Do data reasmpling again on data with fewer dimensions.

Step4: Model Training
Use XGBoost regression algorithm and use gridsearch to tune parameters.
Note that we use r square as evalution metric of CrossValidation.

"""
# load packages
import os
import time
import copy
import datetime
import pickle as p
import numpy as np 
import pandas as pd
pd.options.display.max_columns = 200

from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from script import feature_selection_by_rfe, transformation, evaluation

random_state = 7
model_random_state = 7
closest_station_number = 6
outlier_threshold = 1
target = "PM10"
df_raw = pd.read_csv('data_integration/df_based_AQ_GW.csv')
print('df_raw.shape:', df_raw.shape)
# df_raw.drop(['Unnamed: 0', 'weather_o'], axis=1, inplace=True)
df = df_raw.drop(['CO', 'SO2', 'NO2'], axis=1)
df.drop_duplicates(['utc_time', 'station_id'], inplace=True)
df= df[df['utc_time'] >= str(datetime.datetime(2018, 4, 1, 0, 0))] #only use April 2018 to train the model
df_overall = transformation.feature_engineering(df, target, closest_station_number=closest_station_number)

print(df.shape) #407400
print(df_overall.shape) #353196
print(pd.isnull(df_overall).sum()/len(df_overall))
for col in df_overall:
    print(col)

df_overall_no_other_target = df_overall.drop([t for t in ['PM2.5', 'PM10', 'O3'] if t != target], axis=1)
print('before dropna:', df_overall_no_other_target.shape)
df_no_many_missing = df_overall_no_other_target.dropna(thresh=int(.5*len(df_overall_no_other_target.columns))) #they should be the starting date, so no much previous infomation
print('after dropna:', df_no_many_missing.shape)
X, y = transformation.main_train_transformation(df_no_many_missing, target, random_state=random_state, outlier_threshold=outlier_threshold, n_components_=.98, other_threshold=30, PCA=False)
# X = X.drop([col for col in X if any(t in col for t in ['PM2.5', 'PM10', 'O3'])], axis=1)
print('X shape:', X.shape)

#we need sampling with inclusing the weight of the closer time
time_series = pd.to_datetime(X.merge(df[['utc_time']], left_index=True, right_index=True)['utc_time'])
sample_weights = list(pd.to_timedelta([l - pd.Timestamp('2017-01-01 00:00:00') for l in time_series])/np.timedelta64(1, 'Y')+1)

X_sample = X.sample(frac=.5, random_state=random_state, weights=sample_weights)
y_sample = y[y.index.isin(X_sample.index)].reindex(X_sample.index)

if isinstance(outlier_threshold, str) is False:
    outlier_threshold = str(outlier_threshold)
# print(X.columns)
#%%
#RFE
feature_table_rfe = feature_selection_by_rfe.feature_selection_by_rfe(X_sample, y_sample, [XGBRegressor]
    , cv=3, n_jobs=-1, scoring='r2'
    , random_state=random_state, plot_directory='evaluation/'+target+'/', plot_file_name='_'.join(['rfe', target, str(len(X.columns)), 'outlier', outlier_threshold]), show_plot=True)
print('best number of columns:', feature_table_rfe[1])
pd.options.display.max_rows = 200
print(feature_table_rfe[0])

#%%
feature_table_rfe[1]['final_number'] = 50
with open(os.path.join('evaluation', target, target+'_'+str(len(X.columns))+'_len_table'+'_seed_'+str(random_state)+'_outlier_'+outlier_threshold+'.pickle'), 'wb') as file:
    p.dump(feature_table_rfe, file)

#%%
with open(os.path.join('evaluation', target, target+'_'+str(len(X.columns))+'_len_table'+'_seed_'+str(random_state)+'_outlier_'+outlier_threshold+'.pickle'), 'rb') as file:
    feature_table_rfe = p.load(file)
X_rfe = X[feature_table_rfe[0][feature_table_rfe[0]['XGBRegressor_rank'] <= feature_table_rfe[1]['final_number']].index.tolist()]

print('X shape:', X_rfe.shape)
print('y shape:', y.shape)
# X shape: (18395, 50)
# y shape: (18395,)
X_sample = X_rfe.sample(frac=.4, random_state=random_state, weights=sample_weights)
y_sample = y[y.index.isin(X_sample.index)].reindex(X_sample.index)

print('X_sample shape:', X_sample.shape)
print('y_sample shape:', y_sample.shape)
# X_sample shape: (7358, 50)
# y_sample shape: (7358,)

#%%
param = {
    'learning_rate': [.1]
    , 'booster': ['gbtree']
    , 'subsample': [.2, .3 ,.4, .5]
    , 'n_estimators': [400]
    , 'min_child_weight': [25]
    , 'reg_alpha': [.3, .4, .5]
    , 'reg_lambda': [.1, .2, .3, .4, .5, .6]
    , 'colsample_bytree': [.66]
    , 'max_depth': [5]
}
model = XGBRegressor(
    random_state=model_random_state
    , n_jobs=-1
    , early_stopping_rounds=80
    )
time_series = pd.to_datetime(X_sample.merge(df[['utc_time']], left_index=True, right_index=True)['utc_time'])
X_sample_weights = list(pd.to_timedelta([l - pd.Timestamp('2017-01-01 00:00:00') for l in time_series])/np.timedelta64(1, 'Y')+1)
gridsearch = GridSearchCV(model, param_grid=param, cv=KFold(3, random_state=model_random_state), n_jobs=-1, scoring='r2', return_train_score=True)
t0 = time.time()
gridsearch.fit(X_sample, y_sample, **{'sample_weight': X_sample_weights}) #you may specify sample_weight=weights here
t1 = time.time()

print('time of gridsearch:', round(t1-t0, 2)) #calculate the time for gridsearch
print('best params:', gridsearch.best_params_)
best_position = gridsearch.best_index_
print('best train score:', gridsearch.cv_results_['mean_train_score'][best_position])
print('best train std:', gridsearch.cv_results_['std_train_score'][best_position])
print('best test score:', gridsearch.cv_results_['mean_test_score'][best_position])
print('best test std:', gridsearch.cv_results_['std_test_score'][best_position])
# time of gridsearch: 168.47
# best params: {'booster': 'gbtree', 'colsample_bytree': 0.66, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 25, 'n_estimators': 400, 'reg_alpha': 0.3, 'reg_lambda': 0.2, 'subsample': 0.5}
# best train score: 0.9542144634445969
# best train std: 0.0026327197813804243
# best test score: 0.8121860061388676
# best test std: 0.010038811534079112

#%%
xgbr = XGBRegressor(
    random_state=model_random_state
    , n_jobs=-1
    , early_stopping_rounds=350
    )
best_model = copy.deepcopy(xgbr)
updated_dict = gridsearch.best_params_
updated_dict['learning_rate'] = .01
updated_dict['n_estimators'] = 2000
updated_dict['subsample'] = .25
updated_dict['max_depth'] = 4
updated_dict['min_child_weight'] = 50
best_model.set_params(**updated_dict)
#build final model here
time_series = pd.to_datetime(X.merge(df[['utc_time']], left_index=True, right_index=True)['utc_time'])
X_weights = list(pd.to_timedelta([l - pd.Timestamp('2017-01-01 00:00:00') for l in time_series])/np.timedelta64(1, 'Y')+1)
scores = evaluation.cv_scores(best_model, X_rfe, y, cv=KFold(3, random_state=model_random_state), fit_params={'sample_weight': X_weights}, return_estimator=True)
final_model = scores['estimator'][0]
# train mean of r2: 0.8592183840838218
# train std of r2: 0.006366524558710944
# test mean of r2: 0.8155369094347594
# test std of r2: 0.0260219785886948
# train mean of smape: 0.1676342317310969
# train std of smape: 0.0021556135029841953
# test mean of smape: 0.17545552619389101
# test std of smape: 0.0031718018750788828

#%%
#save the final model
final_model = scores['estimator'][0]
p.dump(final_model, open(os.path.join('model', '_'.join([target, str(model).split('(')[0], 'seed',  str(random_state), 'seed_for_model',  str(model_random_state), 'outlier', outlier_threshold+'_.pickle'])), 'wb'))
