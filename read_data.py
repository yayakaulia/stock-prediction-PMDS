import numpy as np
import pandas as pd
import joblib

def read_data(path, path_ihsg, 
              save_file = True,
              return_file = True,
              set_index = None):


    emiten = pd.read_csv(path, index_col = set_index)
    ihsg = pd.read_csv(path_ihsg, index_col = set_index)
    merged = pd.merge(emiten, ihsg, how='left', on='Date')
    merged['Close+1'] = merged['Close_x'].shift(-1)
    merged = merged.drop(merged.index[len(merged)-1])

    if save_file:
        joblib.dump(merged, "output/merged.pkl")
    
    if return_file:
        return merged



def split_input_output(dataset,
                       target_column,
                       save_file = True,
                       return_file = True):
    
    output_df = dataset[target_column].reset_index(drop=True)
    input_df = dataset.drop([target_column],
                            axis = 1)
    
    if save_file:
        joblib.dump(output_df, "output/output_df.pkl")
        joblib.dump(input_df, "output/input_df.pkl")
    
    if return_file:
        return output_df, input_df

def x_split(input_df, return_file=True, save_file=True):
    X_train = input_df[:int(input_df.shape[0]*0.6)]
    test_val = input_df[int(input_df.shape[0]*0.6):]
    X_valid = test_val[:int(test_val.shape[0]*0.5)]
    X_test = test_val[int(test_val.shape[0]*0.5):]

    if save_file:
        joblib.dump(X_train, "output/X_train.pkl")
        joblib.dump(X_valid, "output/X_valid.pkl")
        joblib.dump(X_test, "output/X_test.pkl")

    if return_file:
        return X_train, X_valid, X_test
        
def y_split(output_df, return_file=True, save_file=True):
    y_train = output_df[:int(output_df.shape[0]*0.6)]
    y_test_val = output_df[int(output_df.shape[0]*0.6):]
    y_valid = y_test_val[:int(y_test_val.shape[0]*0.5)]
    y_test = y_test_val[int(y_test_val.shape[0]*0.5):]
    
    if save_file:
        joblib.dump(y_train, "output/y_train.pkl")
        joblib.dump(y_valid, "output/y_valid.pkl")
        joblib.dump(y_test, "output/y_test.pkl")

    if return_file:
        return y_train, y_valid, y_test