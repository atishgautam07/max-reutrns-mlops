from dataclasses import dataclass
# from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    source_dir: str
    fetchRepo: bool
    calcTickers: bool
    bucketName: str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: str
    data_path: str
    transfData: bool
    bucketName: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: str
    num_trials: int
    cv: int
    data_path: str
    mlflow_uri: str
    hpo_exp_rf: str
    hpo_exp_xgb: str
    trainModel: bool
    bucketName: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: str
    data_path: str
    top_n: int
    ml_uri: str
    hpo_exp_rf: str
    hpo_exp_xgb: str
    exp_name: str
    trainModel: bool
    bucketName: str


@dataclass(frozen=True)
class ModelPredictionConfig:
    root_dir: str
    data_path: str
    ml_uri: str
    hpo_exp_rf: str
    hpo_exp_xgb: str
    exp_name: str
    model_name_xgb: str
    model_name_rf: str
    prediction_name: str
    bucketName: str


@dataclass(frozen=True)
class ModelMonitoringConfig:
    root_dir: str
    data_path: str
    ml_uri: str
    exp_name: str
    model_name_xgb: str
    model_name_rf: str
    prediction_name: str
    DB_URI: str
    TABLE_NAME: str
    bucketName: str