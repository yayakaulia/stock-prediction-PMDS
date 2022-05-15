import pandas as pd
from sklearn.preprocessing import StandardScaler 
import joblib

def make_sma(xdata):
    periode = [5,20,60,120]
    alpha = [0.1, 0.3]
    for i in periode:
        for k in alpha:
            xdata["SMA_", i] = xdata.Close.rolling(i, min_periods=1).mean()
            xdata["dis_sma", i] = xdata["Close"] - xdata["SMA_", i]
            xdata["em_", k] = xdata.Close.ewm(alpha=k, adjust=False).mean()
    xdata.rename(columns = {('SMA_', 5):'SMA_5',
                        ('SMA_', 20):'SMA_20',
                        ('SMA_', 60): 'SMA_60', 
                        ('SMA_', 120): 'SMA_120',
                        ('em_', 0.1): 'em_0.1',
                        ('em_', 0.3): 'em_0.3',
                        ('dis_sma', 5): 'dis_sam_5',
                        ('dis_sma', 20): 'dis_sam_20',
                        ('dis_sma', 60): 'dis_sam_60',
                        ('dis_sma', 120): 'dis_sam_120'}, inplace = True)
    return xdata

def normalization(x_all):
    index = x_all.index
    cols = x_all.columns
    normalizer = StandardScaler()
    normalizer.fit(x_all)        
    normalized = normalizer.transform(x_all)
    normalized = pd.DataFrame(normalized)
    normalized.index = index
    normalized.columns = cols
    return normalized

def making_sma(save_file=True, return_file=True):
    X_train = normalization(make_sma(joblib.load("output/X_train_proceed.pkl")))
    X_valid = normalization(make_sma(joblib.load("output/X_valid_proceed.pkl")))
    X_test = normalization(make_sma(joblib.load("output/X_test_proceed.pkl")))
    
    if save_file:
        joblib.dump(X_train, "output/X_train_final.pkl")
        joblib.dump(X_valid, "output/X_valid_final.pkl")
        joblib.dump(X_test, "output/X_test_final.pkl")
    if return_file:
        X_train, X_valid, X_test