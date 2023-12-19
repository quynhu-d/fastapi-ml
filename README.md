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
3. `predict`: predict for one sample using fitted model (see documentation via swagger)

## Example:
Pass `data/train_data.json` as `data` and `data/svr_config.json` as `model_config_path` to `train` upon query.
