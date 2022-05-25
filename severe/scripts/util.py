from os.path import join
import pandas as pd

def get_right_units_vil(vil):
    """they scaled VIL weird, so this unscales it"""
    tmp = np.zeros(vil.shape)
    idx = np.where(vil <=5)
    tmp[idx] = 0
    idx = np.where((vil>5)*(vil <= 18))
    tmp[idx] = (vil[idx] -2)/90.66
    idx = np.where(vil>18)
    tmp[idx] = np.exp((vil[idx] - 83.9)/38.9)
    return tmp

def unscale_data(X):
    """Unscale the dataset"""
    scaling = { 'wv' : {'mean' : -37.076042, 'std' : 11.884567},
                'ir' : {'mean' : -16.015541, 'std' : 25.805252},
               'vl' : {'mean' : 0.40534842, 'std' : 1.9639382}, 
               'vi' : {'mean' : 0.14749236, 'std' : 0.21128087}
              }
    X_unscaled = X.copy()
    for f in X.columns:
        v = f.split('_')[-1]
        X_unscaled[f] = (X[f] * scaling[v]['std']) + scaling[v]['mean']

    return X_unscaled    
        

# Load the data 
def load_ml_data(path='../datasets', task = 'classification'):
    """Load the sub-SEVIR ML data
    Parameters
    ---------------
    path : path-like string 
        Path to the data 
    task : 'classification' or 'regression'
        Whether to load the classification or 
        regression target labels. 
    
    Returns
    -------------
     List of 2-tuples of X,y for the training, validation, and testing datasets
    """
    y_col = 'label_class' if task == 'classification' else 'label_reg'
    c=0
    data = []
    for mode in ['train', 'val', 'test']:
        df = pd.read_csv(join(path, f'lowres_features_{mode}.csv' ))
        if c == 0:
            features  = [f for f in df.columns if 'label' not in f]
            c+=1
        X = df[features]
        X = unscale_data(X)
        y = df[y_col]
        data.append(X)
        data.append(y)

    return data


# Make a pipeline combining the classification and regression-based models. 

import numpy as np
class CombinedClassifierAndRegressor:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        print(regressor)
        self.regressor = regressor 
    
    def _get_positive_indices(self, X):
        y_pred = self.classifier.predict(X)
        return np.where(y_pred>0)[0]
    
    def fit(self,X,y):
        if isinstance(X, pd.DataFrame):
            X = X.values 
            
        inds = self._get_positive_indices(X)
        
        X_sub = X[inds, :]
        y_sub = y[inds]
                
        self.regressor.fit(X_sub, y_sub)
        
        return self
    
    def predict(self,X):
        inds = self._get_positive_indices(X)
        
        predictions = np.zeros(X.shape[0])
        
        reg_pred = self.regressor.predict(X)
        predictions[inds] = reg_pred[inds]
        
        return predictions 
