import uvicorn
from fastapi import FastAPI, HTTPException
import os

from custom_typing import Data
from custom_typing import LinearRegressionConfig, SVRConfig
from custom_typing import DecisionTreeRegressorConfig
from custom_typing import RandomForestRegressorConfig
from typing import Optional, Union

from model import AVAILABLE_MODELS as models_list, eval_trained_model
from model import get_model, load_model, save_model, delete_model


app = FastAPI(
    title="Student grade prediction",
    openapi_url="/grade_alcohol_api.json"
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

    Returns dictionary with items:
    - **prediction**: grade predictions
    - **mse**: mse score if target is provided
    - **mae**: mae score if target is provided
    """
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    model = load_model(model_path)
    try:
        prediction = model.predict(data.features)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    output = {'predicted grades': prediction.tolist()}
    if data.targets is not None:
        output |= eval_trained_model(model, data, train_mode=False)
    return output


@app.post("/train", summary='Train model on given data.')
async def train(
    data: Data,
    model_type: str,
    params: Union[LinearRegressionConfig, RandomForestRegressorConfig, DecisionTreeRegressorConfig, SVRConfig],
    model_path: Optional[str] = None, cv_eval: Optional[int] = 2
):
    """
    Train regression model.

    Parameters:

    - **data** (Data): data to be fitted on
    - **model_type** (str): model to fit (LinearRegression/DecisionTreeRegressor/RandomForestRegressor/SVR)
    - **params** (LinearRegressionConfig/RandomForestRegressorConfig/DecisionTreeRegressorConfig/SVRConfig):
    model hyperparameters
    - **model_path** (str): path to save model file
    - **cv_eval** (int): number of folds for evaluating

    Returns: dictionary with items:
    - **mse** (float): mse score on train set,
    - **mae** (float): mae score on train set,
    - **cv_neg_mse** (list of float): negated mse scores for each fold,
    - **cv_neg_mae** (list of float): negated mae scores for each fold
    """
    if model_type not in models_list:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_type} not available. Please choose from {models_list}."
        )
    model = get_model(model_type, params)
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


@app.put("/retrain", summary='Retrain previously fitted model on new data.')
async def retrain(
    data: Data,
    model_path: str = '../models/svr.pkl',
    cv_eval: Optional[int] = 2
):
    """
    Retrain regression model.

    Parameters:

    - **data** (Data): data to be fitted on
    - **model_path** (str): path to previously fitted model
    - **cv_eval** (int): number of folds for evaluating

    Returns: dictionary **metrics** with items:

    - **mse** (float): mse score on train set,
    - **mae** (float): mae score on train set,
    - **cv_neg_mse** (list of float): negated mse scores for each fold,
    - **cv_neg_mae** (list of float): negated mae scores for each fold
    """
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    model = load_model(model_path)
    if data.targets is None:
        raise HTTPException(
            status_code=400,
            detail="Target not provided"
        )
    try:
        model.fit(data.features, data.targets)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    save_model(model, model_path)
    metrics = eval_trained_model(model, data, cv=cv_eval)
    return metrics


@app.delete("/delete_model", summary='Delete model.')
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

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8008, reload=True)
