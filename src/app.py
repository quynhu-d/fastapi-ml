import uvicorn
from fastapi import FastAPI, HTTPException
from data import Data
from typing import Optional
from model import AVAILABLE_MODELS as models_list
from model import get_model, load_model, save_model, delete_model, eval_trained_model
import os


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
    return "Available models: " + ', '.join(models_list)


@app.post("/predict", summary='Predict grade.')
async def predict(data: Data, model_path: str = '../models/svr.pkl'):
    """
    Predict grade for students.

    Parameters:

    - **data** (Data): sample features
    - **model_path** (str): path to fitted model
    
    Returns:
    
    - **prediction** (dict): grade predictions
    - **mae, mse** if target is provided
    """
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    model = load_model(model_path)
    prediction = model.predict(data.features)

    if data.targets is not None:
        metrics = eval_trained_model(model, data, train_mode=False)   
    return {'predicted grades': prediction.tolist()} | metrics

@app.post("/train", summary='Train model on given data.')
async def train(
    data: Data,
    model_type: str = 'SVR', 
    model_config_path: str = None,
    # model_config: ModelConfig = ModelConfig(), 
    model_path: Optional[str] = None, cv_eval: Optional[int] = 2
):
    """
    Train regression model.

    Parameters:

    - **data** (Data): data to be fitted on
    - **model_type** (str): model to fit (LinearRegression/DecisionTreeRegressor/RandomForestRegressor/SVR)
    - **model_path** (str): path to save model file
    - **cv_eval** (int): number of folds for evaluating
    
    Returns: dictionary **metrics** with items:
    - **cv_neg_mse** (list of float): negated mse scores for each fold,
    - **cv_neg_mae** (list of float): negated mae scores for each fold,
    - **mse_train** (float): mse score on train set,
    - **mae_train** (float): mae score on train set
    """
    if model_type not in models_list:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_type} not available. Please choose from {models_list}."
        )
    model = get_model(model_type, model_config_path)
    if data.targets is None:
        raise HTTPException(
            status_code=400,
            detail="Target not provided"
        )
    model.fit(data.features, data.targets)
    if model_path:
        save_model(model, model_path)
    metrics = eval_trained_model(model, data, cv=cv_eval)   
    return metrics 

@app.post("/delete_model", summary='Delete model.')
async def delete(
    model_path: str
):
    """
    Delete model at path.

    Parameters:

    - **model_path** (str): path to saved model
    """
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    delete_model(model_path)
    return f"Deleted model {model_path}"
    #TODO: return hyperparameters/print smth


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8008)