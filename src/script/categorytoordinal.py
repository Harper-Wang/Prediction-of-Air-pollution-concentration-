import numpy as np
import pandas as pd

class CategoryToOrdinal():
    """
    Transforming the category column to ordinal column ranked by the target variable.
    
    Parameters
    ----------
    other_threshold : int, optional
        The threshold to group the rare category into one group.
        For example : if other_threshold=15, and 'A', 'B' occur 10 and 12 times in the data. Then these two categories will be grouped as one.

    Attributes
    ----------
    self.stored_cols_dict : dict
        the dictionary store the information of the transformation, the format is as follow:
        {
            col1: {
                'rank_value': {
                    {'A': 2, 'B' : 1}
                },
                'specal_value': {
                    {'mode': 2, 'other':3, 'NA': 4}
                }
            },
            col2: ...
        }

    Notes
    -----
    The function will transform the NA value as well.
    The function will group the rare other cases (< other_threshold) as single group.
    Then the new category (in .transform) will be assigned as the ordinal value of 'other' (in .fix). If there is no 'other' in .fix, the new category will be labled as the mode in .fix.

    """
    def __init__(self, other_threshold=30):
        self.other_threshold = other_threshold

    def fit(self, X, y):
        """
        Fix the CategoryToOrdinal of X by the label y.
        
        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Input data to be fit, where n_samples is the number of samples and n_features is the number of features.
        y : pandas.Series, shape (n_samples, )
            Input target to be fit, it should be ordinal
        """
        stored_cols_dict = {}
        X_merged = pd.concat([X, y], axis=1)
        for col in X.columns.tolist():
            stored_col_dict = {}
            group_by_table = X_merged.groupby(col)[y.name].agg(['mean', 'count']).reset_index(drop=False)
            group_by_table2 = group_by_table[group_by_table['count'] < self.other_threshold]
            group_by_table3 = group_by_table.drop(group_by_table2.index)
            if group_by_table2.empty is False: #add the other (index=-1) if there is rare observation
                group_by_table3.loc[-1] = ['other', group_by_table2['count'].sum(), sum(group_by_table2['count']*group_by_table2['mean'])/group_by_table2['count'].sum()]
            if len(X_merged[X_merged[col].isnull()]) > 0:
                group_by_table3.loc[-2] = ['NA', len(X_merged[X_merged[col].isnull()]), X_merged[X_merged[col].isnull()][y.name].mean()]

            group_by_table3.sort_values('mean', inplace=True)
            group_by_table3['rank'] = range(1, len(group_by_table3)+1)
            stored_col_dict['speical_value'] = {'mode': group_by_table3.sort_values('count')['rank'].iloc[-1]}
            if group_by_table2.empty is False: #record the 'other' in 'spical_value'
                stored_col_dict['speical_value']['other'] = group_by_table3.loc[-1, 'rank']
                group_by_table3.drop(-1, axis=0, inplace=True)
            if len(X_merged[X_merged[col].isnull()]) > 0:
                stored_col_dict['speical_value']['NA'] = group_by_table3.loc[-2, 'rank']
                group_by_table3.drop(-2, axis=0, inplace=True)
            stored_col_dict['rank_value'] = dict(zip(group_by_table3[col], group_by_table3['rank']))
            
            stored_cols_dict[col] = stored_col_dict
        self.stored_cols_dict = stored_cols_dict
        return self

    def transform(self, X):
        """
        Transform the cols in X based on the fixed class and return a new X dataframe.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            Input data to be transformed, where n_samples is the number of samples and n_features is the number of features.
        
        Returns
        -------
        X_transformed : pandas.DataFrame, shape (n_samples, n_features)
            The transformed X pandas.DataFrame
        """
        X_transformed = X.copy()
        for col in X_transformed.columns.tolist():
            if col in self.stored_cols_dict.keys():
                X_transformed[col] = [
                    self.stored_cols_dict[col]['rank_value'][row] if row in self.stored_cols_dict[col]['rank_value'].keys()
                    else self.stored_cols_dict[col]['speical_value']['NA'] if (pd.isnull(row)) and ('NA' in self.stored_cols_dict[col]['speical_value'].keys())
                    else self.stored_cols_dict[col]['speical_value']['other'] if 'other' in self.stored_cols_dict[col]['speical_value'].keys()
                    else self.stored_cols_dict[col]['speical_value']['mode']
                    for row in X_transformed[col]]
        return X_transformed
