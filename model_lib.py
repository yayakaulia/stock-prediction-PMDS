import joblib
import yaml
import numpy as np
from sklearn import metrics

f = open("params.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

def read_data(params):
    x_train = joblib.load(params['DUMP_TRAIN'])
    y_train = joblib.load(params['Y_PATH_TRAIN'])
    x_valid = joblib.load(params['DUMP_VALID'])
    y_valid = joblib.load(params['Y_PATH_VALID'])

    return x_train, y_train, x_valid, y_valid

def model_ridge():
    param_dist = {'alpha': [0.1, 0.25, 0.5, 0.75], 'solver': ['svd', 'cholesky', 'lsqr', 'sag']}
    return param_dist

def model_lasso():
    param_dist = {'alpha': np.random.uniform(0.01,3,1000)}
    return param_dist

def model_rf():
    param_dist = {"n_estimators": [100, 250, 500, 1000]}
    return param_dist

def model_svr():
    param_dist = {'C': [0.25, 0.5, 1, 1.25]}
    return param_dist

def get_params():
    ridge = model_ridge()
    lasso = model_lasso()
    rf = model_rf()
    svr = model_svr()

    paramet = eval(joblib.load(params['MODEL_NAME']))
    return paramet

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def validation_score(x_valid, y_valid, model_fitted):
    
    # Report default
    y_predicted = model_fitted.predict(x_valid)
    mae, mse, rmse, r2_square = evaluate(y_valid, y_predicted)
    score = {'mae':mae, 'mse':mse, 'rmse':rmse, 'r2': r2_square}
    return score
