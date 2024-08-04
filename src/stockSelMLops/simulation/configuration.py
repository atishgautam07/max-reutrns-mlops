from google.cloud import storage
from common import read_yaml, create_directories, create_gcs_directories
from pathlib import Path
from config_entity import (DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPredictionConfig, ModelMonitoringConfig, SimulateStrategyConfig)


class ConfigurationManager:
    def __init__(self, bucket_name , gcs_file_path):
        
        gcs_file_path = '/'.join(gcs_file_path.split('/')[3:])

        self,
        self.bucket_name = bucket_name

        print ('Setting up configs for pipeline.')
        config_file_path = Path("config.yaml") #Path('src/stockSelMLops/ingestion/research/config_ingest.yaml')
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
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config['model_trainer']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config['root_dir'],
            num_trials=config['num_trials'],
            cv = config['cv'],
            data_path = config['data_path'],
            mlflow_uri = config['mlflow_uri'],
            hpo_exp_rf = config['hpo_exp_rf'],
            hpo_exp_xgb = config['hpo_exp_xgb'],
            trainModel = config['trainModel'],
            bucketName=self.bucket_name 
        )

        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config['root_dir'],
            data_path=config['data_path'],
            top_n=config['top_n'],
            ml_uri=config['ml_uri'],
            hpo_exp_rf=config['hpo_exp_rf'],
            hpo_exp_xgb=config['hpo_exp_xgb'],
            exp_name=config['exp_name'],
            trainModel = config['trainModel'],
            bucketName=self.bucket_name
        )

        return model_evaluation_config


    def get_model_prediction_config(self) -> ModelPredictionConfig:
        config = self.config['model_prediction']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])
        
        model_prediction_config = ModelPredictionConfig(
            root_dir=config['root_dir'],
            data_path=config['data_path'],
            ml_uri=config['ml_uri'],
            hpo_exp_rf=config['hpo_exp_rf'],
            hpo_exp_xgb=config['hpo_exp_xgb'],
            exp_name=config['exp_name'],
            model_name_xgb=config["model_name_xgb"],
            model_name_rf=config["model_name_rf"],
            prediction_name=config["prediction_name"],
            bucketName=self.bucket_name
        )

        return model_prediction_config


    def get_model_monitoring_config(self) -> ModelMonitoringConfig:
        config = self.config['model_monitoring']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])
        
        model_monitoring_config = ModelMonitoringConfig(
            root_dir=config['root_dir'],
            data_path=config['data_path'],
            ml_uri=config['ml_uri'],
            exp_name=config['exp_name'],
            model_name_xgb=config["model_name_xgb"],
            model_name_rf=config["model_name_rf"],
            prediction_name=config["prediction_name"],
            # DB_URI=config["DB_URI"],
            INSTANCE_CONNECTION_NAME=config["INSTANCE_CONNECTION_NAME"],
            DB_USER=config["DB_USER"],
            DB_PASS=config["DB_PASS"],
            DB_NAME=config["DB_NAME"],
            TABLE_NAME=config["TABLE_NAME"],
            bucketName=self.bucket_name
        )

        return model_monitoring_config
    

    def get_simulate_strategy_config(self) -> SimulateStrategyConfig:
        config = self.config['simulate_strategy']

        if config['root_dir'].startswith('gs://'):
            self.create_gcs_paths(config['root_dir'])
        else:
            create_directories([config['root_dir']])
        
        simulate_strategy_config = SimulateStrategyConfig(
            root_dir=config['root_dir'],
            data_path=config['data_path'],
            rsi=config["rsi"],
            prob_thres=config["prob_thres"],
            LE1=config["LE1"],
            LE2=config["LE2"],
            SL=config["SL"],
            PP=config["PP"],
            window=config["window"],
            starting_corpus=config["starting_corpus"],
            bucketName=self.bucket_name
        )

        return simulate_strategy_config

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

