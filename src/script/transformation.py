import os
import pickle as p
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from script import categorytoordinal

def shifting(df, shift_col, groupby_col=None, shift_number=1):
    """Shift the shift_col with previous value (shift_number)
    
    Parameters
    ----------
    df : pandas.DataFrame
        input data
    shift_col : str
        the col to be shifted
    groupby_col : str, optional
        the groupby col
    shift_number : int, optional
        the number of shift of previous row
    
    Returns
    -------
    pandas.DataFrame
        The output data with added the shifted column
    """
    
    if groupby_col is None:
        df.loc[:, shift_col+'(t-'+str(shift_number)+')'] = df[shift_col].shift(shift_number)
    else:
        df.loc[:, shift_col+'(t-'+str(shift_number)+')'] = df.groupby(groupby_col)[shift_col].shift(shift_number)
    return df

def feature_engineering(df, target, groupby_col='station_id', closest_station_number=5, drop=True):
    """Generate the features including Information from previous timestamp, nearby station, Arithmetic operation
    
    Parameters
    ----------
    df : pandas.DataFrame
        input data
    target : str
        the target col
    groupby_col : str, optional
        the groupby col for shifting function
    closest_station_number : int, optional
        the number of closest station we use for each station
    drop : boolean, optional
        whether to drop the row with missing target, it is not applicable for final prediction
    
    Returns
    -------
    pandas.DataFrame
        output data
    
    See Also
    --------
    shifting
    """
    df0 = df.copy()
    df = df.copy()
    target_list = ['PM2.5', 'PM10', 'O3']
    df.loc[:, 'utc_time'] = pd.to_datetime(df['utc_time'], errors='coerce') 
    df = df.sort_values(['station_id', 'utc_time']) # sort all values by station and time
    # df.loc[:, 'year'] = df['utc_time'].dt.year
    df.loc[:, 'month'] = df['utc_time'].dt.month
    # df.loc[:, 'day'] = df['utc_time'].dt.day
    df.loc[:, 'weekday'] = df['utc_time'].dt.weekday
    df.loc[:, 'hour'] = df['utc_time'].dt.hour
    # holidays = pd.read_csv('data_integration/holidays.csv')
    # holidays = pd.to_datetime(holidays['holiday'], errors='coerce')
    # a = list()
    # for i in df['utc_time'].dt.date.values:
    #     if i in holidays.dt.date.values:
    #         a.append(1)
    #     else:
    #         a.append(0)
    # df.loc[:,'holiday'] = pd.Series(a)

    numeric_columns = ['PM2.5', 'PM10', 'O3', 'temperature_g', 'pressure_g', 'humidity_g', 'wind_direction_g', 'wind_speed_g', 'temperature_o', 'pressure_o', 'humidity_o', 'wind_direction_o', 'wind_speed_o']
    categorical_columns = ['weather_g', 'weather_o', 'weather']
    
    #shift the time
    for t in range(1, 9): #shift the target
        df = shifting(df, shift_col=target, groupby_col='station_id', shift_number=t)
    for t in range(1, 6): #shift other tathe
        for col in [t for t in target_list if t != target]+['wind_speed_g', 'temperature_g', 'pressure_g', 'wind_direction_g']:
            if col in df:
                df = shifting(df, shift_col=col, groupby_col='station_id', shift_number=t)
    for t in range(1, 4):
        for col in ['humidity_g', 'weather']:
            if col in df:
                df = shifting(df, shift_col=col, groupby_col='station_id', shift_number=t)
    
    #na treatment for numeric columns
    for col in df:
        if any(num_col in col for num_col in numeric_columns):
            if col != target:
                df.loc[:, col] = df.groupby(groupby_col)[col].apply(lambda group: group.interpolate())

    #na treatment for categorical columns
    for col in df:
        if any(cate_col in col for cate_col in categorical_columns):
            df.loc[:, col] = df.groupby(groupby_col)[col].fillna(method='ffill')
    if drop:
        df = df[df[target].notnull()]
    # add closest location
    df_station_id = pd.read_csv('data_integration/station_id_closest.csv')
    df_station_id.rename(columns={'station_id': 'station_id_r'}, inplace=True)
    df = df.reset_index().merge(df_station_id,  how='left', left_on='station_id', right_on='station_id_r').drop('station_id_r', axis=1).set_index('index')

    df_closest = df[[col for col in ['station_id', 'utc_time']+numeric_columns+[t+'(t-1)' for t in target_list] if col in df]]
    for i in range(1, closest_station_number+1):
        rename_dict = {l: l+'_closest'+str(i) for l in numeric_columns+[t+'(t-1)' for t in target_list]}
        rename_dict['station_id'] = 'station_id_r'
        rename_dict['utc_time'] = 'utc_time_r'
        df_closest_reanme = df_closest.rename(columns=rename_dict)
        df = df.reset_index().merge(df_closest_reanme, how='left', left_on=['closest_station'+str(i), 'utc_time'], right_on=['station_id_r', 'utc_time_r']).drop(['station_id_r', 'utc_time_r'], axis=1).set_index('index')
    df.drop([col for col in df if 'closest_station' in col], axis=1, inplace=True) #drop the col containing the 'closest_station' wording
    #if time gap with the previous record (by group) > 5, the missing value will be overwritten filled by nearby station in shifting, instead of the interpolate() method
    # df = df.assign(output=df.groupby('station_id')['utc_time'].apply(lambda x: x - x.shift()).astype('timedelta64[h]')) #create a column called 'output' which records the time gap
    # for col in numeric_columns:
    #     #for non shifted columns
    #     closest_cols = [col+'_closest'+str(j) for j in range(1, closest_station_number+1)]
    #     df.loc[:, col] = pd.Series(np.where(df['output'] < 5, df[col], df[closest_cols].mean(axis=1))).fillna(df[col]) #fillna is for if closest columns are not available all

    #     #for shifted columns
    #     if fill_shifted_cols_by_closest:
    #         for i in range(1, shift_number+1): #i is the shift number
    #             shifted_col = col+'(t-'+str(i)+')'
    #             shifted_closest_cols = [shifted_col+'_closest'+str(j) for j in range(1, closest_station_number+1)]
    #             df.loc[:, shifted_col] = pd.Series(np.where(df['output'] < 5, df[shifted_col], df[closest_cols].mean(axis=1))).fillna(df[shifted_col]) #fillna is for if closest columns are not available all
    # df.drop('output', axis=1, inplace=True)

    drop_closest_list = []
    for col in df:
        if '_closest' in col:
            if (any(target in col for target in ['PM2.5', 'PM10', 'O3', 'wind_direction_g', 'wind_speed_g']) is True) & (any(shift in col for shift in ['(t-1)', '(t-2)']) is True): #don't drop PM2.5(t_1)_closest... columns
                pass
            else:
                drop_closest_list.append(col)
    df.drop(drop_closest_list, axis=1, inplace=True)

    #generate the feature of percentage change of the (pollution(t-1) - pollution(t-2))/pollution(t-2)
    for col in target_list:
        if col in df:
            for t in range(1, 3): # t=1 or t=2
                df.loc[:, col+'(t-'+str(t)+') percent change'] = (df[col+'(t-'+str(t)+')'] - df[col+'(t-'+str(t+1)+')'])/df[col+'(t-'+str(t+1)+')']
    
    #add multiplication
    for t in range(1, 3):
        for i in range(0, len(target_list)):
            if target_list[i] in df.columns:
                for j in range(i+1, len(target_list)):
                    if target_list[j] in df.columns:
                        df.loc[:, target_list[i]+'(t-'+str(t)+')'+'*'+target_list[j]+'(t-'+str(t)+')'] = df[target_list[i]+'(t-'+str(t)+')']*df[target_list[j]+'(t-'+str(t)+')']

    df.drop([col for col in ['grid_id', 'utc_time', "longitude", "latitude"] if col in df], axis=1, inplace=True)
    # df.drop([col for col in target_list if col !=target], axis=1, inplace=True)
    return df

def main_train_transformation(df, target, outlier_threshold='3IQR', PCA=False, n_components_=.98, other_threshold=30, random_state=1):
    """Main transformation function for training, it includes categorical variables to ordinal
    
    Parameters
    ----------
    df : pandas.DataFrame
        input data
    target : str
        the target col
    outlier_threshold : str or float, optional
        [description] (the default is '3IQR', which means the target values greater than Q3 + 3*IQR are outlier )
    PCA : bool, optional
        whether to apply PCA or not
    n_components : int, float, None or string
        Number of components to keep. if n_components is not set all components are kept:
    other_threshold : int, optional
        The threshold to group the rare category into one group.
        For example : if other_threshold=15, and 'A', 'B' occur 10 and 12 times in the data. Then these two categories will be grouped as one.
    random_state : integer, RandomState instance or None, optional 
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. (the default is None)
    
    Returns
    -------
    pandas.DataFrame
        output data
    """

    print('Before outlier:\n', df[target].describe())
    if outlier_threshold != '3IQR':
        df_target_no_outlier = df[df[target] <= df[target].quantile(outlier_threshold)]
    else:
        df_target_no_outlier = df[df[target] <= df[target].quantile(.75) + 3*(df[target].quantile(.75) - df[target].quantile(.25))]
    print('After outlier:\n', df_target_no_outlier[target].describe())

    #categorical transformation
    cto = categorytoordinal.CategoryToOrdinal(other_threshold=other_threshold)
    categorical_col = ['station_id', 'station_class'] + [col for col in df_target_no_outlier.columns if 'weather' in col]
    cto.fit(df_target_no_outlier[categorical_col], df_target_no_outlier[target])
    df_transformed = cto.transform(df_target_no_outlier)
    with open('input/categorytoordinal_'+target+'_'+str(other_threshold)+'_seed_'+str(random_state)+'.pickle', "wb") as f_out:
        p.dump(cto, f_out)

    #Fill the missing value per station
    imp = SimpleImputer(missing_values=np.nan, strategy='mean') #even if there is no np.nan, we still do the na value treatment, since testing dataset may contains NA value.
    imp.fit(df_transformed)
    df_fillna = pd.DataFrame(imp.fit_transform(df_transformed), index=df_transformed.index, columns=df_transformed.columns)
    with open('input/imputer_'+target+'_seed_'+str(random_state)+'.pickle', "wb") as f_out:
        p.dump(imp, f_out)

    df_no_target = df_fillna.drop(target, axis=1)

    #standardize the data
    sc = StandardScaler()
    df_scalez = pd.DataFrame(data=sc.fit_transform(df_no_target), index=df_no_target.index, columns=df_no_target.columns)
    with open('input/standardscaler_'+target+'_'+str(other_threshold)+'_seed_'+str(random_state)+'.pickle', "wb") as f_out:
        p.dump(sc, f_out)
    print('shape of scalez:', df_scalez.shape)

    #PCA
    if PCA:
        pca = PCA(n_components=n_components_)
        df_pca = pd.DataFrame(data=pca.fit_transform(df_scalez), index=df_scalez.index, columns=['pca_'+str(i+1) for i in range(0, pca.n_components_)])
        with open('input/pca_'+target+'_'+str(n_components_)+'_seed_'+str(random_state)+'.pickle', "wb") as f_out:
            p.dump(pca, f_out)
        X = df_pca.copy()
        X.to_csv(os.path.join('data_integration', '_'.join([target, 'X', 'train', str(len(X.columns)), 'len', 'seed', str(random_state)])+'_withPCA'+'.csv'), index_label=False)
    else:
        X = df_scalez.copy()
        X.to_csv(os.path.join('data_integration', '_'.join([target, 'X', 'train', str(len(X.columns)), 'len', 'seed', str(random_state)])+'_noPCA'+'.csv'), index_label=False)
    y = df_transformed[target].reindex(X.index)
    y.to_csv(os.path.join('data_integration', '_'.join([target, 'y', 'train', 'seed', str(random_state)])+'.csv'), index_label=False)
    #Print info
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('finite test:', np.all(np.isfinite(X))) #should be True
    print('na test', np.any(np.isnan(X))) #should be False
    return X, y
