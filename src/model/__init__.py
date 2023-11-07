from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List
from ..data import Item, process_data
import json


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

AVAILABLE_MODELS = ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR"]

# class ModelConfig(BaseModel):
#     fit_intercept: Optional[bool] = None

    # max_depth: Optional[int] = None
    # max_features: Optional[float] = None
    # max_leaf_nodes: Optional[int] = None
    # max_samples: Optional[int] = None
    # min_impurity_decrease: Optional[int] = None
    # min_samples_leaf: Optional[int] = None
    # min_samples_split: Optional[int] = None
    # min_weight_fraction_leaf: Optional[float] = None

    # n_estimators: Optional[int] = None

    # C: Optional[float] = None
    # cache_size: Optional[int] = None
    # coef0: Optional[float] = None
    # degree: Optional[int] = None
    # epsilon: Optional[float] = None
    # gamma: Optional[str] = None
    # kernel: Optional[str] = None
    # max_iter: Optional[int] = None
    # shrinking: Optional[bool] = None
    # tol: Optional[float] = None

# class LinearRegressionConfig(ModelConfig):
#     fit_intercept: Optional[bool]

# class DecisionTreeRegressorConfig(ModelConfig):
#     max_depth: Optional[int]
#     max_features: Optional[float]
#     max_leaf_nodes: Optional[int]
#     max_samples: Optional[int]
#     min_impurity_decrease: Optional[int]
#     min_samples_leaf: Optional[int]
#     min_samples_split: Optional[int]
#     min_weight_fraction_leaf: Optional[float]

# class RandomForestRegressorConfig(DecisionTreeRegressorConfig):
#     n_estimators: Optional[int]

# class SVRConfig(ModelConfig):
#     C: Optional[float]
#     cache_size: Optional[int]
#     coef0: Optional[float]
#     degree: Optional[int]
#     epsilon: Optional[float]
#     gamma: Optional[str]
#     kernel: Optional[str]
#     max_iter: Optional[int]
#     shrinking: Optional[bool]
#     tol: Optional[float]

def get_model(
    model_type: str, 
    model_config_path: str = None
    # model_config: ModelConfig = ModelConfig()
):
    if model_config_path:
        with open(model_config_path, 'r') as f:
            model_config_dict = json.load(f)
            model_config_dict = {param: val for param, val in model_config_dict.items() if val}
    else:
        model_config_dict = {}
    # model_config_dict = {param: val for param, val in model_config.dict().items() if val}
    # print(model_config_dict)
    if model_type not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_type} not available. Please choose from {AVAILABLE_MODELS}."
        )
    model = eval(model_type)(**model_config_dict)
    return model