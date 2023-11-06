from fastapi import FastAPI, HTTPException
from src.data import Item
from typing import Optional
from src.predict import predict
app = FastAPI()


@app.get("/")
def index():
    return {
        "Dataset url": "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data"
    }

@app.get("/list_models")
def list_models():
    models_list = []
    with open('src/model/list_models.txt', 'r') as f:
        models_list = f.read().split('\n')
    return "Available models: " + ', '.join(models_list)


@app.post("/predict")
async def predict_for_item(item: Item, model_path: str = './models/svr.pkl'):
    result = predict(item, model_path)  
    return {'predicted grade': result[0]}

@app.post("/train")
async def train(item: Item, model_path: str = './models/svr.pkl'):
    mae = 0
    mse = 0
    return {'Trained': "", "MAE": mae, "MSE": mse}