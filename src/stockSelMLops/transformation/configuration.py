from google.cloud import storage
from common import read_yaml, create_directories, create_gcs_directories
from pathlib import Path
from config_entity import (DataIngestionConfig, DataTransformationConfig)


class ConfigurationManager:
    def __init__(self, bucket_name , gcs_file_path):
        
        gcs_file_path = '/'.join(gcs_file_path.split('/')[3:])

        self,
        self.bucket_name = bucket_name

        print ('Setting up configs for ingestion.')
        config_file_path = Path("config_ingest.yaml") #Path('src/stockSelMLops/ingestion/research/config_ingest.yaml')
        self.download_from_gcs(gcs_file_path, config_file_path)
        print (f"Downloaded configs from cloud - {gcs_file_path}")

        self.config = read_yaml(config_file_path)

        if self.config['artifacts_root'].startswith('gs://'):
            self.create_gcs_paths(self.config['artifacts_root'])
        else:
            create_directories([self.config['artifacts_root']])



    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config['root_dir'],
            source_dir=config['source_dir'],
            fetchRepo=config['FETCH_REPO'],
            calcTickers=config['calcTickers'],
            bucketName=self.bucket_name
        )
        return data_ingestion_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])

        data_transformation_config = DataTransformationConfig(
            root_dir=config['root_dir'],
            data_path=config['data_path'],
            transfData=config['TRANSFORM_DATA'],
            bucketName=self.bucket_name 
        )

        return data_transformation_config


    def download_from_gcs(self, gcs_file_path, local_file_path):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(gcs_file_path)
        blob.download_to_filename(local_file_path)
        print(f"File {gcs_file_path} downloaded to {local_file_path}.")

    def create_gcs_paths(self, gcs_path):
        # bucket_name = gcs_path.split('/')[2]
        prefix = '/'.join(gcs_path.split('/')[3:])
        create_gcs_directories(self.bucket_name, [prefix])       

