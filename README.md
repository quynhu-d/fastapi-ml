# FastAPI for student grade prediction

Данг Куинь Ньы (Нина), tg: @quynhu_d

Dataset taken from [kaggle](https://www.kaggle.com/code/biancacarvalho/a2-icd-student-alcohol-consumption). 

Objective: predict student grade based (regression).

## To start:

Create virtual environment, install required packages:

```bash
python -m venv fastml_env
source fastml_env/bin/activate
pip install -r requirements.txt
```

## Run API:

```bash
cd src
uvicorn app:app --reload --host 0.0.0.0 --port 8008
```
or
```bash
cd src
python app.py
```

API with Swagger Documentation can be accessed via http://localhost:8008/docs#/.

## Functionality:

1. [`list_models`](http://localhost:8008/list_models) List available models.
2. `train`: train regression model (see documentation via swagger)
3. `predict`: predict using fitted model (see documentation via swagger)
4. `retrain`: retrain previously fitted model (see documentation via swagger)
5. `delete_model`: delete existing model checkpoint

## Example:
Pass `data/train_data.json` as `data` and `examples/model_configs/svr_config.json` as `params` to `train` upon query.

**All parameters must be filled in** (can be None), see LinearRegressionConfig/RandomForestRegressorConfig/DecisionTreeRegressorConfig/SVRConfig specification in Schemas and/or example values in `examples/model_configs`.