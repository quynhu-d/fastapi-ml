# FastAPI for student grade prediction

## To start:

Create virtual environment, install required packages:

```bash
python -m venv fastml_env
source fastml_env/bin/activate
pip install -r requirements.txt
```

## Run API:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8008
```

API with Swagger Documentation can be accessed via http://localhost:8008/docs#/.

## Functionality:

1. [`list_models`](http://localhost:8008/list_models) List available models.
2. `train`: train regression model (see documentation via swagger)
3. `predict`: predict for one sample using fitted model (see documentation via swagger)

## Example:
Pass `data/train_data.json` as `data` and `data/svr_config.json` as `model_config_path` to `train` upon query.
