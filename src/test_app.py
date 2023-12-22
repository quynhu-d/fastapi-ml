from app import app, minio_saver
from fastapi.testclient import TestClient
from fastapi import status
from sklearn.linear_model import LinearRegression


client = TestClient(app=app)

mock_model = LinearRegression()
mock_features = [[1, 2, 3], [4, 5, 6]]
mock_targets = [1, 2]
mock_data = {'features': mock_features, 'targets': mock_targets}
mock_model.fit(mock_features, mock_targets)


def mock_save_to_minio(data, object_name, bucket_name, metadata=None):
    return f"Mock saving to minio: {bucket_name}/{object_name}."


def mock_load_from_minio(object_name, bucket_name):
    if bucket_name == 'models':
        return mock_model
    else:
        return f"Mock loading from minio: {bucket_name}/{object_name}."


def test_index():
    response = client.get('/')
    assert response.status_code == status.HTTP_200_OK
    assert "Dataset url" in response.json()


def test_app_save_data(mocker):
    mock_dataset_name = "test_data.json"
    mocker.patch.object(minio_saver, 'save_to_minio', mock_save_to_minio)
    response = client.post(
        f'/save_data?object_name={mock_dataset_name}',
        json=mock_data
    )
    assert response.status_code == status.HTTP_200_OK


def test_app_train(mocker):
    mocker.patch.object(minio_saver, 'save_to_minio', mock_save_to_minio)
    mock_params = {'fit_intercept': True}
    model_type = "LinearRegression"
    model_path = "test_lr.json"
    response = client.post(
        f'/train?model_type={model_type}&model_path={model_path}',
        json={'data': mock_data, 'params': mock_params}
    )
    assert response.status_code == status.HTTP_200_OK
    assert 'mse' in response.json()
    assert response.json()['Minio message'] == f"Mock saving to minio: models/{model_path}."


def test_app_predict(mocker):
    mocker.patch.object(minio_saver, 'save_to_minio', mock_save_to_minio)
    mocker.patch.object(minio_saver, 'load_from_minio', mock_load_from_minio)
    model_path = "test_lr.json"
    response = client.post(
        f'/predict?model_path={model_path}',
        json=mock_data
    )
    assert response.status_code == status.HTTP_200_OK
    assert 'mse' in response.json()
