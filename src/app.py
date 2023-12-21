import uvicorn
from fastapi import FastAPI, HTTPException

from custom_typing import Data
from custom_typing import LinearRegressionConfig, SVRConfig
from custom_typing import DecisionTreeRegressorConfig
from custom_typing import RandomForestRegressorConfig
from typing import Optional, Union

from model import AVAILABLE_MODELS as models_list, eval_trained_model
from model import get_model
# from model import load_model, save_model, delete_model

from minio.error import S3Error
from minio_utils import MinioSaver
minio_saver = MinioSaver(host="minio", port='9000', user='admin', password='admin1234')

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


@app.get("/list_saved_models", summary='List available trained models.')
def list_saved_models():
    return "Available trained models: " + ', '.join(minio_saver.list_objects_in_bucket('models'))


@app.get("/list_saved_datasets", summary='List available datasets.')
def list_saved_datasets():
    return "Available datasets: " + ', '.join(minio_saver.list_objects_in_bucket('datasets'))


@app.post("/save_data")
async def save_data(data: Data, object_name: str):
    """
    Save data to minio.

    Parameters:

    - **data** (Data): data to save
    - **object_name** (str): path to save data
    """
    minio_mssg = minio_saver.save_to_minio(data, object_name, bucket_name='datasets')
    return minio_mssg


@app.post("/predict", summary='Predict grade.')
async def predict(data: Union[Data, str], model_path: str = 'svr.pkl'):
    """
    Predict grade for students.

    Parameters:

    - **data** (Data or str): data to be fitted on, if str, data is loaded from minio
    - **model_path** (str): path to fitted model (loaded from minio)

    Returns dictionary with items:
    - **prediction**: grade predictions
    - **mse**: mse score if target is provided
    - **mae**: mae score if target is provided
    """
    # if not os.path.exists(model_path):
    #     raise HTTPException(status_code=404, detail="Model not found")
    # model = load_model(model_path)
    try:
        model = minio_saver.load_from_minio(object_name=model_path, bucket_name='models')
    except S3Error as s3e:
        raise HTTPException(status_code=404, detail=str(s3e))
    if isinstance(data, str):
        try:
            data = minio_saver.load_from_minio(data, 'datasets')
        except S3Error as s3e:
            raise HTTPException(status_code=404, detail=str(s3e))
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
    data: Union[Data, str],
    model_type: str,
    params: Union[LinearRegressionConfig, RandomForestRegressorConfig, DecisionTreeRegressorConfig, SVRConfig],
    model_path: Optional[str] = None, cv_eval: Optional[int] = 2
):
    """
    Train regression model.

    Parameters:

    - **data** (Data or str): data to be fitted on, if str, data is loaded from minio
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
    if isinstance(data, str):
        try:
            data = minio_saver.load_from_minio(data, 'datasets')
        except S3Error as s3e:
            raise HTTPException(status_code=404, detail=str(s3e))
    if data.targets is None:
        raise HTTPException(
            status_code=400,
            detail="Target not provided"
        )
    model.fit(data.features, data.targets)
    if model_path:
        minio_mssg = minio_saver.save_to_minio(model, model_path, bucket_name='models')
        # save_model(model, model_path)
    metrics = eval_trained_model(model, data, cv=cv_eval)
    return metrics | {'Minio message': minio_mssg}


@app.put("/retrain", summary='Retrain previously fitted model on new data.')
async def retrain(
    data: Union[Data, str],
    model_path: str = 'svr.pkl',
    cv_eval: Optional[int] = 2
):
    """
    Retrain regression model.

    Parameters:

    - **data** (Data or str): data to be fitted on, if str, data is loaded from minio
    - **model_path** (str): path to previously fitted model
    - **cv_eval** (int): number of folds for evaluating

    Returns: dictionary **metrics** with items:

    - **mse** (float): mse score on train set,
    - **mae** (float): mae score on train set,
    - **cv_neg_mse** (list of float): negated mse scores for each fold,
    - **cv_neg_mae** (list of float): negated mae scores for each fold
    """
    # if not os.path.exists(model_path):
    #     raise HTTPException(status_code=404, detail="Model not found")
    # model = load_model(model_path)
    try:
        model = minio_saver.load_from_minio(object_name=model_path, bucket_name='models')
    except S3Error as s3e:
        raise HTTPException(status_code=404, detail=str(s3e))
    if isinstance(data, str):
        try:
            data = minio_saver.load_from_minio(data, 'datasets')
        except S3Error as s3e:
            raise HTTPException(status_code=404, detail=str(s3e))
    if data.targets is None:
        raise HTTPException(
            status_code=400,
            detail="Target not provided"
        )
    try:
        model.fit(data.features, data.targets)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    minio_mssg = minio_saver.save_to_minio(model, model_path, bucket_name='models')
    # save_model(model, model_path)
    metrics = eval_trained_model(model, data, cv=cv_eval)
    return metrics | {'Minio message': minio_mssg}


@app.delete("/delete_model", summary='Delete model.')
async def delete_model(
    model_path: str
):
    """
    Delete model at path.

    Parameters:

    - **model_path** (str): path to saved model
    """
    minio_mssg = minio_saver.delete_from_minio(object_name=model_path, bucket_name='models')
    return {'Minio message': minio_mssg}
    # if not os.path.exists(model_path):
    #     raise HTTPException(status_code=404, detail="Model not found")
    # delete_model(model_path)
    # return f"Deleted model {model_path}"


@app.delete("/delete_data", summary='Delete data.')
async def delete_data(
    data_path: str
):
    """
    Delete data at path.

    Parameters:

    - **data_path** (str): path to saved data
    """
    minio_mssg = minio_saver.delete_from_minio(object_name=data_path, bucket_name='datasets')
    return {'Minio message': minio_mssg}


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8008, reload=True)
