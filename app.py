from fastapi import FastAPI, HTTPException, Body
from src.data import Item
from typing import Dict, List, Optional
from src.predict import predict
from src.model import AVAILABLE_MODELS as models_list, get_model
from src.train import train_model, eval_trained_model
import os
import pickle

app = FastAPI(
    title = "Student grade prediction",
    openapi_url = "/grade_alcohol_api.json"
)

@app.get("/", summary='Index page.')
def index():
    return {
        "Dataset url": "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data"
    }

@app.get("/list_models", summary='List available regression models.')
def list_models():
    # models_list = []
    # with open('src/model/list_models.txt', 'r') as f:
    #     models_list = f.read().split('\n')
    return "Available models: " + ', '.join(models_list)


@app.post("/predict", summary='Predict grade for one sample.')
async def predict_for_item(item: Item, model_path: str = './models/svr.pkl'):
    """
    Predict grade for one student.

    Parameters:

    - **item** (Item): sample features
    - **model_path** (str): path to fitted model
    
    Returns:
    
    - **prediction** (dict): grade prediction
    """

    prediction = {'predicted grade': predict(item, model_path)}
    return prediction

@app.post("/train", summary='Train model on given data.')
async def train(
    data: Dict[str, Item], cols: Optional[List[str]] = None, 
    model_type: str = 'SVR', 
    model_config_path: str = None,
    # model_config: ModelConfig = Body(embed=True, media_type = 'config/json'), 
    model_path: Optional[str] = None, cv_eval: Optional[int] = 2
):
    """
    Train regression model.

    Parameters:

    - **data** (list of Item): data to be fitted on
    - **cols** (list of str or None): names of features to use for training
    - **model_type** (str): model to fit (LinearRegression/DecisionTreeRegressor/RandomForestRegressor/SVR)
    - **model_path** (str): path to save model file
    - **cv_eval** (int): number of folds for evaluating
    
    Returns: dictionary with items:
    TODO
    - **cv_neg_mse**: [],
    - "cv_neg_mae": [],
    -"mse_train": 0,
    -"mae_train": 0
    """
    model = get_model(model_type, model_config_path)
    # model = get_model(model_type)
    model, (x, y) = train_model(model, data, cols)
    if model_path:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    metrics = eval_trained_model(model, x, y, cv=cv_eval)   
    return metrics 