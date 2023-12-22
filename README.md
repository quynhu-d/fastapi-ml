# FastAPI for student grade prediction

Данг Куинь Ньы (Нина), tg: @quynhu_d

Dataset taken from [kaggle](https://www.kaggle.com/code/biancacarvalho/a2-icd-student-alcohol-consumption). 

Objective: predict student grade based (regression).

## HW1

Check branch `hw1_checkpoint`.

### To start:

Create virtual environment, install required packages:

```bash
python -m venv fastml_env
source fastml_env/bin/activate
pip install -r requirements.txt
```

### Run API:

```bash
cd src
uvicorn app:app --reload --host 0.0.0.0 --port 8008
```
or
```bash
cd src
python app.py
```

## HW2

Check branch `hw2_checkpoint`.
Added minio functionality, tried dvc.

App docker image: `quynhud/grade-prediction-app:latest`.

To run application:
```bash
docker-compose up -d
```
Minio s3 at http://localhost:9001/.

## HW3

Unit tests in `src/test_app.py`
```bash
pytest src/test_app.py
```
Loading and saving functions for s3 are mocked.

CI with Github Actions: `.github/workflows/ci.yml`:
- Builds and pushes Docker images to Docker Hub **only upon merge requests**
- Flake8 linter (allows lines up to 100 characters)

## Documentation

API with Swagger Documentation can be accessed via http://localhost:8008/docs#/.

## Functionality:

1. [`list_models`](http://localhost:8008/list_models) List available models.
2. [`list_saved_models`](http://localhost:8008/list_saved_models): List saved trained models.
3. [`list_saved_datasets`](http://localhost:8008/list_saved_datasets): List saved datasets.
4. `save_data`: Save datasets.
5. `train`: train regression model (see documentation via swagger)
6. `predict`: predict using fitted model (see documentation via swagger)
7. `retrain`: retrain previously fitted model (see documentation via swagger)
8. `delete_model`: delete existing model checkpoint
9. `delete_data`: delete existing data

## Example:
Pass `data/train_data.json` as `data` and `examples/model_configs/svr_config.json` as `params` to `train` upon query.

**All parameters must be filled in**, see LinearRegressionConfig/RandomForestRegressorConfig/DecisionTreeRegressorConfig/SVRConfig specification in Schemas and/or example values in `examples/model_configs`.
