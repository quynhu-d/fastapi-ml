from fastapi import HTTPException
from .model import BaseModel
import pandas as pd

from typing import Dict, List, Optional
from .data import Item, process_data

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

def train_model(model: BaseModel, data: Dict[str, Item], cols: Optional[List[str]] = None):
    x, y, _ = process_data(data, cols)
    if y is None:
        raise HTTPException(
            status_code=404,
            detail="Target not provided"
        )
    model.fit(x, y)
    return model, (x, y)

def eval_trained_model(model: BaseModel, x: pd.DataFrame, y:pd.Series, cv: int = 5):
    y_pred = model.predict(x)
    metrics = {
        'cv_neg_mse': cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error').tolist(),
        'cv_neg_mae': cross_val_score(model, x, y, cv=cv, scoring='neg_mean_absolute_error').tolist(),
        'mse_train': mse(y, y_pred),
        'mae_train': mae(y, y_pred)
    }
    return metrics