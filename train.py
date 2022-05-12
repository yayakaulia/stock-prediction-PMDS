from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
import yaml
import pandas as pd
import numpy as np
import joblib
import model_search
import model_lib

ts_cv = TimeSeriesSplit(
    n_splits=5,
    max_train_size=None,
)
f = open("params.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

def training_model():

    #read data
    x_train, y_train, x_valid, y_valid = model_lib.read_data(params)

    ridge = [Ridge(random_state=42), model_lib.model_ridge]
    lasso = [Lasso(random_state=42), model_lib.model_lasso]
    rf = [RandomForestRegressor(random_state=42), model_lib.model_rf]
    svr = [LinearSVR(random_state=42), model_lib.model_svr]

    # define model
    model = eval(joblib.load(params['MODEL_NAME']))[0]
    alpa = eval(joblib.load(params['MODEL_NAME']))[1]

    scoring = 'neg_root_mean_squared_error'

    # define search
    search = GridSearchCV(model, alpa(), scoring=scoring, n_jobs=-1, cv=ts_cv)

    # execute search
    result = search.fit(x_train, y_train)
    validation_result = model_lib.validation_score(x_valid, y_valid, result)
    
    print('Best Score: %s' % validation_result)


    # Dump model name
    joblib.dump(model, f'output/model_name.pkl')
    # Dump best model estimator with best param
    joblib.dump(result.best_params_, 'output/best_estimator.pkl')
    # summarize result

    return result