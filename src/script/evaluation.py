import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))

def cv_scores(model, X, y, cv, scoring={'r2': make_scorer(r2_score), 'smape': make_scorer(smape)}, fit_params=None, return_estimator=True):
    scores = cross_validate(model, X, y, cv=cv, return_train_score=True, n_jobs=-1, scoring=scoring, fit_params=fit_params, return_estimator=return_estimator)
    for key in scoring.keys():
        for t in ['train', 'test']:
            print(t+' mean of '+key+':', scores[t+'_'+key].mean())
            print(t+' std of '+key+':', scores[t+'_'+key].std())
    return scores
