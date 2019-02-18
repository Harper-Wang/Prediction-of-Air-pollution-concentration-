# -*- coding: utf-8 -*-
'''
<copyright file='feature_selection_by_rfe.py' company='GenLife'>
Copyright (c) 2017 Gen.Life Limited All rights reserved.
<date>2018-03-20</date>
</copyright>
'''
#%%
import os
import pathlib
import warnings
import pandas as pd
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score, make_scorer, r2_score

def to_png(fig, plot_path=None, plot_directory=None, plot_file_name=None, picture_size_w=None, picture_size_h=None, dgi=900, show_plot=False, close_plot=True):
    """
    Save the plot path 
    
    Parameters
    ----------
    fig : object
        matplotlib fig object
    plot_path : str, optional
        the full path of the file (including the file name, no need to specifiy '.png'). If it does not exist, the plot_path will be created by plot_directory and plot_file_name
    plot_directory : str, optional
        the directory of the file
    plot_file_name : str, optional
        The file name in the plot_directory (no need to specifiy '.png'). Noted that the slash in file_name will be converted as dot.
    picture_size_w : None or float, optional
        the width of the picture
    picture_size_h : None or float, optional
        the height of the picture
    dgi : int, optional
        The quality of the photo
    show_plot : bool, optional
        whether to show the plotting
    close_plot : bool, optional
        whether to close the plotting if show_plot is True
    """
    try:    
        fig.tight_layout()
    except: #ValueError: left cannot be >= right
        pass
    if (plot_path is None) & ((plot_directory is not None) & (plot_file_name is not None)):
        plot_path = os.path.join(plot_directory, (plot_file_name.replace('/', '.')+'.png').replace('.png.png', '.png'))

    if (picture_size_w is not None) & (picture_size_h is not None):
        fig.set_size_inches(picture_size_w, picture_size_h)

    if plot_path is not None:
        pathlib.Path(plot_path.rsplit('/', 1)[0]+'/').mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(plot_path.rsplit('/', 1)[0], plot_path.rsplit('/', 1)[1].replace('+', '')), dgi=dgi, bbox_inches='tight')
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()

def _concat_result(concat_result, X, ranking, test):
    """
    It outputs the concatenated result.
    
    Parameters
    ----------
    concat_result : pandas.DataFrame
        The table that want to be concat.
    X : pandas.DataFrame
        Get the column names.
    ranking : list
        the list of ranking variables of n_features
    test : str
        the model short name
    
    Returns
    -------
    pandas.DataFrame
        The output concatenated dataframe, where the index is the var (feature).
    """
    model_result = pd.DataFrame({test+'_score': ranking}, index=X.columns)
    model_result[test+'_rank'] = model_result[test+'_score'].rank(method='average').astype(int)
    concat_result = pd.concat([concat_result , model_result[[test+'_rank']]], axis=1)
    concat_result['avg_rank'] = concat_result['avg_rank'] + model_result[test+'_rank']
    return concat_result

def _plot_feature_selected_with_performance(grid_scores, model_name='', scoring_name='', 
    optimal_number_of_feature=None, fontsize=12, picture_size_scale=1,
    show_plot=True, plot_path=None, plot_file_name=None, plot_directory=None):
    """
    It plots the graph of number of features versus the cross-validation result

    Parameters
    ----------
    grid_scores : list
        The cross-validation score list based on the models with different number of features
    model_name : str, optional
        the model name of underlying estimator model used
    scoring_name : str, optional
        the scoring name of the cross validation used to evaluate the peformance of the model
    optimal_number_of_feature : integer or None, optional
        The optimal number of feature.
    fontsize : int, optional
        It controls the fontsize of title, x-label, y-label. Remark that the fontsize will be automaically adjusted by the picture size.
    picture_size_scale : float, optional
        the user-defined scale of the picture, where the base is 1
    show_plot : bool, optional
        whether to show the plotting
    plot_path : str, optional
        the full path of the file (including the file name, no need to specifiy '.png'). If it does not exist, the plot_path will be created by plot_directory and plot_file_name
    plot_file_name : str, optional
        the user-defined file name, no need to write '.png'. If it is None, then the file name will be generated automatically.
    plot_directory : str, optional
        If it is None, then the graph will not be saved as output. It should be a directory.
    """

    scale = picture_size_scale**(0.3) #the scale of the word shown in the picture

    #set the title
    fig, ax = plt.subplots()
    plt.plot(range(1, len(grid_scores) + 1), grid_scores)
    plt.xlabel('Number of features selected', fontsize=fontsize*scale)
    plt.ylabel('Cross validation score (' + scoring_name.replace('_', ' ') + ')', fontsize=fontsize*scale)
    plt.title('Model peformance of ' + model_name,
        y = 1 +0.01, fontsize=(fontsize+1)*scale, fontweight='bold')
    #set grid
    plt.grid(True)

    if optimal_number_of_feature != None:
        plt.plot(optimal_number_of_feature, grid_scores[optimal_number_of_feature-1], marker='o', label='Optimal number of feature')
    #set the legend
    plt.legend(loc='best', fontsize=fontsize*scale)    

    try:    
        plt.tight_layout()
    except: #ValueError: left cannot be >= right
        pass
    
    fig = plt.gcf() #get current figure
    fig.set_size_inches(15*picture_size_scale, 9*picture_size_scale) #addjust the size
    to_png(fig=fig, plot_path=plot_path, plot_directory=plot_directory, plot_file_name='rfe' if plot_file_name is None else plot_file_name, show_plot=show_plot) #save the png

def _feature_selection_by_rfe(X, y, sklearn_model, param=None, cv=3, scoring=None, n_jobs=3, verbose=0, random_state=None):
    """
    calculate the rfe per model.

    Parameters
    ----------
    X : array-like or sparse matrix, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values (integers in classification, real numbers in regression) For classification, labels must correspond to classes.
    sklearn_model_list : sklearn.model
        list of sklearn model
    param : dict, optional
        The user defined list for param of your input sklearn_model_list. If it is None, the model will use the default params (see Note).
    scoring : sklearn.metices, optional
        slearn.metrics of classification or regression, such as: roc_auc_score, accuracy, f1_score etc.
        If it is None, then it will assign accuracy_score (for classification) or r2_score (for regression).
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
        For integer/None inputs, if y is binary or multiclass, sklearn.model_selection.StratifiedKFold is used. If the estimator is a classifier or if y is neither binary nor multiclass, sklearn.model_selection.KFold is used.
    n_jobs : int, optional
        Number of cores to run in parallel while fitting across folds. Defaults to 1 core. If n_jobs=-1, then number of jobs is set to number of cores.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    random_state : integer, RandomState instance or None, optional 
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. (the default is None)
    
    Returns
    -------
    tuple (pandas.DataFrame, dict)
        - the rfe model with n_features_to_select=1, which provides the importance ranking of variables.
        - the rfecv model with, which provides the model performace.
    
    Notes
    -----
    The default param of the following models:
        - RandomForestClassifier: {'n_estimators': 200, 'min_samples_leaf': .05, 'criterion': 'gini', 'max_features': 'auto', 'class_weight': 'balanced_subsample', 'n_jobs': n_jobs}
        - GradientBoostingClassifier: {'n_estimators': 100, 'min_samples_leaf': .05, 'criterion': 'friedman_mse', 'max_features': 'auto', 'learning_rate': .01, 'subsample': .9}
        - XGBClassifier: {'n_estimators': 100, 'learning_rate': .01, 'min_child_weight': min(round(len(X)*.05), 30), 'subsample': .9, 'n_jobs': n_jobs}
        - LinearSVC: {'C': .1, 'multi_class': 'ovr', 'max_iter': 1000}
        - LogisticRegression: {'C': .1, 'multi_class': 'ovr', 'max_iter': 500}
        - RandomForestRegressor: {'n_estimators': 300, 'min_samples_leaf': .05, 'criterion': 'mse', 'max_features': 'auto', 'n_jobs': n_jobs}
        - GradientBoostingRegressor: {'n_estimators': 150, 'min_samples_leaf': .05, 'loss': 'ls', 'criterion': 'friedman_mse', 'max_features': 'auto', 'learning_rate': .01, 'subsample': .9}
        - XGBRRegressor: {'n_estimators': 150, 'learning_rate': .01, 'min_child_weight': min(round(len(X)*.05), 30), 'subsample': .9, 'n_jobs': n_jobs}
        - LinearSVR: {'C': .1, 'loss': 'epsilon_insensitive', 'max_iter': 1000}
        - Ridge: {'alpha': .1, 'max_iter': 1000, 'slover': 'auto'}
    """
    model_name = str(sklearn_model)
    #classifier
    if 'RandomForestClassifier' in model_name:
        default_param = {'n_estimators': 200, 'min_samples_leaf': .05, 'criterion': 'gini', 'max_features': 'auto', 'class_weight': 'balanced_subsample', 'n_jobs': n_jobs}
    elif 'GradientBoostingClassifier' in model_name:
        default_param = {'n_estimators': 100, 'min_samples_leaf': .05, 'criterion': 'friedman_mse', 'max_features': 'auto', 'learning_rate': .01, 'subsample': .9}
    elif 'XGBClassifier' in model_name:
        default_param = {'n_estimators': 100, 'learning_rate': .01, 'min_child_weight': min(round(len(X)*.05), 30), 'subsample': .9, 'n_jobs': n_jobs}
    elif 'LinearSVC' in model_name:
        default_param = {'C': .1, 'multi_class': 'ovr', 'max_iter': 1000}
    elif 'LogisticRegression' in model_name:
        default_param = {'C': .1, 'multi_class': 'ovr', 'max_iter': 500}
    #regression
    elif 'RandomForestRegressor' in model_name:
        default_param = {'n_estimators': 300, 'min_samples_leaf': .05, 'criterion': 'mse', 'max_features': 'auto', 'n_jobs': n_jobs}
    elif 'GradientBoostingRegressor' in model_name:
        default_param = {'n_estimators': 150, 'min_samples_leaf': .05, 'loss': 'ls', 'criterion': 'friedman_mse', 'max_features': 'auto', 'learning_rate': .01, 'subsample': .9}
    elif 'XGBCRegressor' in model_name:
        default_param = {'n_estimators': 150, 'learning_rate': .01, 'min_child_weight': min(round(len(X)*.05), 30), 'subsample': .9, 'n_jobs': n_jobs}
    elif 'LinearSVR' in model_name:
        default_param = {'C': .1, 'loss': 'epsilon_insensitive', 'max_iter': 1000}
    elif 'Ridge' in model_name:
        default_param = {'alpha': .1, 'max_iter': 1000, 'slover': 'auto'}
    else: #for unknown model
        default_param = dict()

    if isinstance(param, dict) is False:
        param = dict()
    for k, v in default_param.items():
        if k not in param.keys():
            param[k] = v
    param['random_state'] = random_state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe_model = RFE(estimator=sklearn_model(**param), n_features_to_select=1, verbose=verbose).fit(X, y.values.ravel())
        RFECV_model = RFECV(estimator=sklearn_model(**param), cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose).fit(X, y.values.ravel())
    return (rfe_model, RFECV_model)

def feature_selection_by_rfe(X, y, sklearn_model_list, param_list=None, 
    scoring=None, cv=3, n_jobs=7, verbose=0, random_state=None,
    fontsize=12, picture_size_scale=1, 
    show_plot=True, plot_file_name=None, plot_directory=None):
    """
    It applies the sklean RFE method to test how the number of features affect the performance of models, then outputs the ranking of the features and the png result.
    
    Parameters
    ----------
    X : array-like or sparse matrix, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values (integers in classification, real numbers in regression) For classification, labels must correspond to classes.
    sklearn_model_list : list of sklearn.model
        list of sklearn model
    param_list : list of dict, optional
        The user defined list for param of your input sklearn_model_list. If it is None, the model will use the default params (see Note).
    scoring : sklearn.metices, optional
        slearn.metrics of classification or regression, such as: roc_auc_score, accuracy, f1_score etc.
        If it is None, then it will assign accuracy_score (for classification) or r2_score (for regression).
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
        For integer/None inputs, if y is binary or multiclass, sklearn.model_selection.StratifiedKFold is used. If the estimator is a classifier or if y is neither binary nor multiclass, sklearn.model_selection.KFold is used.
    n_jobs : int, optional
        Number of cores to run in parallel while fitting across folds. Defaults to 1 core. If n_jobs=-1, then number of jobs is set to number of cores.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    random_state : integer, RandomState instance or None, optional 
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. (the default is None)
    fontsize : int, optional
        It controls the fontsize of title, x-label, y-label. Remark that the fontsize will be automaically adjusted by the picture size.
    picture_size_scale : float, optional
        the user-defined scale of the picture, where the base is 1
    show_plot : bool, optional
        whether to show the plotting
    plot_file_name : str, optional
        the extra user-defined file name, no need to write '.png'. If it is None, then the file name will be generated automatically.
        if plot_file_name = 'abc' and the test is 'rfc', the output name will be 'feature_selection_rfc_abc.png'
    plot_directory : str, optional
        If it is None, then the graph will not be saved as output. It should be a directory.
    
    Raises
    ------
    ValueError
        If your test is not one of the ['rfc', 'gbc', 'lsvc', 'logr'].
    
    Returns
    -------
    tuple (pandas.DataFrame, dict)
        - the summary dataframe shows all the model results
        - dictionary indicates the best n_features under each model
    png
        the output pngs of n_features selected versus model peformance.

    See Also
    --------
    sklearn.feature_selection.RFE : http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
    sklearn.feature_selection.RFECV : http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

    Notes
    -----
    The function is able to work for the multi-classification problem:
        - sklearn.ensemble.RandomForestClassifier implements it direcly
        - sklearn.ensemble.GradientBoostingClassifier implements ovr only
        - sklearn.svm.LinearSVC: ('ovr' or 'crammer_singer')
        - sklearn.linear_model.LogisticRegression: ('ovr' or 'multinomial')
    Noted that for multi-class, many scoring functions are not appliciable, but they should be able to work on 'ovr'.

    The default param of the following models:
        - RandomForestClassifier: {'n_estimators': 200, 'min_samples_leaf': .05, 'criterion': 'gini', 'max_features': 'auto', 'class_weight': 'balanced_subsample', 'n_jobs': n_jobs}
        - GradientBoostingClassifier: {'n_estimators': 100, 'min_samples_leaf': .05, 'criterion': 'friedman_mse', 'max_features': 'auto', 'learning_rate': .01, 'subsample': .9}
        - XGBClassifier: {'n_estimators': 100, 'learning_rate': .01, 'min_child_weight': min(round(len(X)*.05), 30), 'subsample': .9, 'n_jobs': n_jobs}
        - LinearSVC: {'C': .1, 'multi_class': 'ovr', 'max_iter': 1000}
        - LogisticRegression: {'C': .1, 'multi_class': 'ovr', 'max_iter': 500}
        - RandomForestRegressor: {'n_estimators': 300, 'min_samples_leaf': .05, 'criterion': 'mse', 'max_features': 'auto', 'n_jobs': n_jobs}
        - GradientBoostingRegressor: {'n_estimators': 150, 'min_samples_leaf': .05, 'loss': 'ls', 'criterion': 'friedman_mse', 'max_features': 'auto', 'learning_rate': .01, 'subsample': .9}
        - XGBCRegressor: {'n_estimators': 150, 'learning_rate': .01, 'min_child_weight': min(round(len(X)*.05), 30), 'subsample': .9, 'n_jobs': n_jobs}
        - LinearSVR: {'C': .1, 'loss': 'epsilon_insensitive', 'max_iter': 1000}
        - Ridge: {'alpha': .1, 'max_iter': 1000, 'slover': 'auto'}

    Examples
    --------
    >>> import pytest
    >>> import pandas as pd
    >>> from sklearn.preprocessing import Imputer, StandardScaler
    >>> from genlib.ml import feature_selection_by_rfe as fsr
    >>> from genlib import utils
    >>> data = utils.read_csv_to_df('data/prudential_scaled_data.csv')
    >>> #sample datat to increase the speed for just testing
    >>> data = data.sample(frac=.005, replace=True, random_state=42)
    >>> target = 'Response'

    >>> #select the '_1' or '_2'
    >>> columns_end_with_1 = []
    >>> for column in list(data):
    ...    if column[-2:] == '_1':
    ...        columns_end_with_1.append(column)
    ...    else:
    ...        pass
    >>> columns_end_with_1 = columns_end_with_1 + ['Wt', 'BMI', 'Ht']
    >>> X_with_missing = data[columns_end_with_1].select_dtypes(exclude=['object'])
    >>> Imputer_mean = Imputer(strategy='mean')
    >>> scaler = StandardScaler()
    >>> X = pd.DataFrame(scaler.fit_transform(Imputer_mean.fit_transform(X_with_missing)), columns=X_with_missing.columns, index=X_with_missing.index)
    >>> y = data[[target]]
    >>> #TODO: rewrite it later
    >>> feature_table_rfe = fsr.feature_selection_by_rfe(X, y, test_list=['rfc', 'gbc', 'lsvc', 'logr'],
    ...    random_state=42)

    >>> print(feature_table_rfe[0])
                            avg_rank  rfc_rank  gbc_rank  lsvc_rank  logr_rank
    BMI                      1.00               1               1                1                1
    Wt                       2.75               2               3                2                4
    Medical_History_1        3.50               3               4                4                3
    Employment_Info_1        4.50               4               2                6                6
    Insurance_History_1      4.50               6               7                3                2
    InsuredInfo_1            6.75               7               8                7                5
    Medical_Keyword_1        7.50               9               9                5                7
    Ht                       7.50               5               5               10               10
    Family_Hist_1            7.75               8               6                9                8
    Product_Info_1           9.25              10              10                8                9

    >>> print(feature_table_rfe[1])
        {'rfc': 9, 'gbc': 6, 'lsvc': 1, 'logr': 2}
    """
    if param_list is None:
        param_list = [None]*len(sklearn_model_list)
    concat_result_table = pd.DataFrame({'avg_rank': [0]*len(list(X))}, index=X.columns)
    best_n_feature_dict = {}
    for sklearn_model, param  in zip(sklearn_model_list, param_list):
        model_name = str(sklearn_model).split('.')[-1].replace("'>", '')
        if scoring is None:
            if ('regressor' in model_name.lower()) | ('linearregression' in model_name.lower())| ('lasso' in model_name.lower())| ('ridge' in model_name.lower())| ('svr' in model_name.lower()):
                scoring=make_scorer(r2_score)
            else:
                scoring=make_scorer(accuracy_score)
                
        (rfe_model, RFECV_model) = _feature_selection_by_rfe(X, y, sklearn_model, param, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        concat_result_table = _concat_result(concat_result_table, X, rfe_model.ranking_, model_name)
        best_n_feature_dict[model_name] = RFECV_model.n_features_

        if (show_plot is True) | (plot_directory is not None):
            s = str(scoring)
            _plot_feature_selected_with_performance(grid_scores=RFECV_model.grid_scores_, model_name=model_name,
                scoring_name=s[s.find('(')+1:s.find(')')], #collecting the str name of scoring
                optimal_number_of_feature=RFECV_model.n_features_, fontsize=fontsize, picture_size_scale=picture_size_scale,
                plot_file_name='_'.join(filter(None, [plot_file_name.replace('.png', '') if plot_file_name is not None else plot_file_name, model_name])),
                show_plot=show_plot, plot_directory=plot_directory)
    concat_result_table['avg_rank'] = concat_result_table['avg_rank']/len(sklearn_model_list)
    return (concat_result_table.sort_values('avg_rank'), best_n_feature_dict)
