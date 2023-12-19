import os
import pickle
from typing import Union
from custom_typing import *

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

AVAILABLE_MODELS = ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR"]

def get_model(
    model_type: str, 
    params: Union[LinearRegressionConfig, RandomForestRegressorConfig, DecisionTreeRegressorConfig, SVRConfig]
):
    model_config_dict = params.model_dump()
    if model_type == 'LinearRegression':
        model = LinearRegression(**model_config_dict)
    if model_type == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(**model_config_dict)
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(**model_config_dict)
    if model_type == 'SVR':
        model = SVR(**model_config_dict)
    return model

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def delete_model(model_path: str):
    os.remove(model_path)

def eval_trained_model(model, data:Data, train_mode: bool = True, cv: int = 5):
    prediction = model.predict(data.features)
    metrics = {
        'mse': mse(data.targets, prediction),
        'mae': mae(data.targets, prediction)
    }
    if train_mode:
        metrics['cv_neg_mse'] = cross_val_score(
            model, data.features, data.targets, cv=cv, scoring='neg_mean_squared_error'
        ).tolist(),
        metrics['cv_neg_mae'] = cross_val_score(
            model, data.features, data.targets, cv=cv, scoring='neg_mean_absolute_error'
        ).tolist(),

    return metrics