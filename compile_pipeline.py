from kfp import compiler, dsl
from kfp.dsl import pipeline, component, Artifact, Dataset, Input, Metrics, Model, Output, InputPath, OutputPath
from kfp.components import load_component_from_file
from google.cloud import aiplatform

PROJECT_ID = "<project-id>"
BUCKET_NAME = "<bucket-name>"
REGION = "<region-name>"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"


# Load the component from the YAML file
ingestion_op = load_component_from_file('config/ingestion_component.yaml')
transformation_op = load_component_from_file('config/transformation_component.yaml')
# model_training_op = load_component_from_file('model_training_component.yaml')
rf_model_training_op = load_component_from_file('config/rf_model_training_component.yaml')
xgb_model_training_op = load_component_from_file('config/xgb_model_training_component.yaml')
model_evaluation_op = load_component_from_file('config/model_evaluation_component.yaml')
model_prediction_op = load_component_from_file('config/model_prediction_component.yaml')
model_monitoring_op = load_component_from_file('config/model_monitoring_component.yaml')
simulate_strategy_op = load_component_from_file('config_copy/simulate_strategy_component.yaml')


@dsl.pipeline(
    name='Data Ingestion, Transformation, and Model Training, Eval, Predict Pipeline',
    description='A pipeline that performs data ingestion, transformation, and model training with hyperparameter optimization.'
)
def ml_pipeline(config_path: str, bucket_name: str):
    
    # Ingestion step, runs after MLflow server starts
    ingestion_step = ingestion_op(config_path=config_path, bucket_name=bucket_name)
    # Transformation step, runs after ingestion
    transformation_step = transformation_op(config_path=config_path, bucket_name=bucket_name).after(ingestion_step)
    
    # # Model training step, runs after transformation
    # model_training_step = model_training_op(config_path=config_path, bucket_name=bucket_name).after(transformation_step)

    # XGB Model training step, runs after RF training
    xgb_model_training_step = xgb_model_training_op(config_path=config_path, bucket_name=bucket_name).after(transformation_step)
    # RF Model training step, runs after transformation
    rf_model_training_step = rf_model_training_op(config_path=config_path, bucket_name=bucket_name).after(xgb_model_training_step)
    # Model evaluation step, runs after training
    model_evaluation_step = model_evaluation_op(config_path=config_path, bucket_name=bucket_name).after(rf_model_training_step)
    # Model prediction step, runs after evaluation
    model_prediction_step = model_prediction_op(config_path=config_path, bucket_name=bucket_name).after(model_evaluation_step)
    # Model monitoring step, runs after prediction
    model_monitoring_step = model_monitoring_op(config_path=config_path, bucket_name=bucket_name).after(model_prediction_step)
    # Simulate strategy
    simulate_strategy_step = simulate_strategy_op(config_path=config_path, bucket_name=bucket_name).after(model_prediction_step)


    # Compile the pipeline
compiler.Compiler().compile(ml_pipeline, 'config/ml_pipeline.json')