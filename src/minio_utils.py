from minio import Minio
import io
from typing import Union, Optional
from custom_typing import Data
from model import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR
import pickle


class MinioSaver:
    def __init__(
            self,
            host='127.0.0.1', port='9000',
            user='admin', password='admin1234'
    ):
        minio_client = Minio(
            f"{host}:{port}",
            access_key=user,
            secret_key=password,
            secure=False
        )
        if not minio_client.bucket_exists('datasets'):
            minio_client.make_bucket('datasets')
        if not minio_client.bucket_exists('models'):
            minio_client.make_bucket('models')
        self.minio_client = minio_client

    def load_from_minio(
        self,
        object_name: str, bucket_name: str
    ):
        response = self.minio_client.get_object(
            bucket_name=bucket_name, object_name=object_name
        )
        return pickle.loads(response.data)

    def save_to_minio(
        self,
        data: Union[
            Data,
            SVR, LinearRegression, DecisionTreeRegressor, RandomForestRegressor
        ],
        object_name: str, bucket_name: str, metadata: Optional[dict] = None
    ):
        byte_data = pickle.dumps(data)
        response = self.minio_client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(byte_data),
            length=len(byte_data),
            metadata=metadata
        )
        return_message = f'Created {response.bucket_name}/{response.object_name} object; '
        return_message += f"etag: {response.etag}, version-id: {response.version_id}."
        return return_message

    def delete_from_minio(
        self, object_name: str, bucket_name: str
    ):
        self.minio_client.remove_object(
            bucket_name=bucket_name,
            object_name=object_name
        )
        return f'Deleted {bucket_name}/{object_name} object.'

    def list_objects_in_bucket(self, bucket_name: str):
        object_names = [i.object_name for i in self.minio_client.list_objects(bucket_name)]
        return object_names
