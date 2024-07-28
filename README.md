# Max Returns MLOps
### End-to-End MLOps project for predicting top daily stocks to invest in to optimize 5-day return.

## Overview
This repository contains an end-to-end MLOps pipeline that predicts the top daily stocks to invest in to optimize a 5-day return. The project leverages various financial data sources, performs data ingestion and transformation, and trains machine learning models using hyperparameter optimization. The final pipeline runs on Vertex AI, with each component containerized and managed via Google Artifact Registry.

## Repository Structure
 - src: Contains different components of the pipeline.
    - Each folder within src contains a configuration manager class that reads and manages configuration from the config_pred.yaml file.
    - Each component has its own Pipfile and Dockerfile.
    - Docker images of each component are pushed to Artifact Registry and used as Kubeflow components in the Vertex AI pipeline.
 - config: Contains YAML configuration files for different components.
    - main.py: Runs the final end-to-end pipeline.

## Pipeline Stages
### Data Ingestion (data_ingestion.py):
 - Downloads and processes financial data.
 - Saves data as Parquet files.
### Data Transformation (data_transformation.py):
 - Adds technical indicators and combines data into a single DataFrame.
### Model Training (model_trainer.py):
 - Trains Random Forest and XGBoost models.
 - Uses Hyperopt for hyperparameter optimization.
 - Logs models and metrics to MLflow.
### Model Evaluation (model_evaluation.py):
 - Evaluates models based on precision scores.
 - Selects and registers the best model with MLflow.
### Prediction (model_predict.py):
 - Uses the registered model for final predictions (top 10 stocks) stored as csv file in /model_preds folder.

## Usage
 - Clone the repository.
 - Set up Google Cloud SDK and authenticate.
 - Configure the config_pred.yaml file with the necessary parameters.
 - Build and push Docker images for each component to Google Artifact Registry.
 - Run the main.py script to execute the end-to-end pipeline.

## Running the Pipeline
 1. Update config_pred.yaml file with path (gcs bucket paths) and other parameters. Copy data_artifacts folder to the bucket.
 2. For individual folders within src:
    - build and push docker image
    ```bash
    docker build -t your-region-docker.pkg.dev/your-project-id/pipeline_stage:latest .
    docker push your-region-docker.pkg.dev/your-project-id/pipeline_stage:latest
    ```
 3. create a vertex wrokbench instance and copy the config folder.
 4. Update yaml files in config folder with image info from previous step.
 5. Update main.py with gcp project id, bucket name, regio name etc.
 6. Run mlflow server - 
    - create postgres db in gcp SQL with required permissions
    - run mlflow server command in the vertex ai terminal
    ```bash
    mlflow server --backend-store-uri postgresql://<usrname>:<password>@<privaeIP>:<port>/<dbname> --default-artifact-root <gcs-bucket-location> --host 0.0.0.0 --port 5000
    ```
 7. Run main.py
 8. Track mlflow experiments at the adress mentioned in config_pred.yaml
The main.py script orchestrates the entire pipeline, running each stage sequentially.

## Deployment
Docker images for each component are built and pushed to Google Artifact Registry. These images are then used as Kubeflow components in the Vertex AI pipeline.

## Requirements
Google Cloud SDK
Vertex AI
Kubeflow Pipelines
Docker

## License
This project is licensed under the MIT License.