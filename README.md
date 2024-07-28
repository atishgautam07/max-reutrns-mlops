# Max Returns MLOps
### End-to-End MLOps project for predicting top daily stocks to invest in to optimize 5-day return.

## Overview
This repository contains an end-to-end MLOps pipeline that predicts the top daily stocks to invest in to optimize a 5-day return. The project leverages various financial data sources, performs data ingestion and transformation, and trains machine learning models using hyperparameter optimization. The final pipeline runs on Vertex AI using kubeflow pipelines, with each component containerized and managed via Google Artifact Registry.

## Repository Structure
 - src: Contains different components of the pipeline.
    - Each folder within src contains a configuration manager class that reads and manages configuration from the config_pred.yaml file.
    - Each component has its own Pipfile and Dockerfile.
    - Docker images of each component are pushed to Artifact Registry and used as Kubeflow components in the Vertex AI pipeline.
 - config: Contains YAML configuration files for different components.
    - compile_pipeline.py: Compiles all hte components of the pipeline. 
    - run_pipeline.py: Runs the final end-to-end pipeline.

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
 - Create a Vertex AI workbench instance and postgres db in Google cloud SQL for mlflow.
 - Clone the repository.
 - Set up Google Cloud SDK and authenticate.
 - Configure the config_pred.yaml file with the necessary parameters.
 - Build and push Docker images for each component to Google Artifact Registry.
 - Run the compile_pipeline and run_pipeline script to execute the end-to-end pipeline using Vertex AI SDK.

## Running the Pipeline
 1. Update config_pred.yaml file with path (gcs bucket paths) and other parameters. Copy data_artifacts folder to the bucket.
 2. For individual folders within src:
    - build and push docker image
    ```bash
    docker build -t your-region-docker.pkg.dev/your-project-id/pipeline_stage:latest .
    docker push your-region-docker.pkg.dev/your-project-id/pipeline_stage:latest
    ```
 3. Update yaml files in config folder with image info from previous step.
 4. Run mlflow server on a vertex workbench instance- 
    - create postgres db in gcp SQL with required permissions
    ```bash
    mlflow server --backend-store-uri postgresql://<usrname>:<password>@<privaeIP>:<port>/<dbname> --default-artifact-root <gcs-bucket-location> --host 0.0.0.0 --port 5000
    ```
 5. Update compile_pipeline and run_pipeline scripts with gcp project id, bucket name, regio name etc.
 6. Run config_pipeline using vertex AI SDK
    Pipeline is compiled into a JSON format that can be uploaded and executed in Vertex AI.
    ```bash
    python config_pipeline.py
    ```
 7. Upload the compiled pipeline JSON to a Google Cloud Storage bucket.
    ```bash 
    gsutil cp ml_pipeline.json gs://bucket-name/pipeline_root/config/ml_pipeline.json
    ```
 8. Use the Vertex AI SDK to create and run the pipeline job.
    ```bash
    python run_pipeline.py
    ```
 9.  Track pipeline job on GCP. 
 10. Track mlflow experiments at the adress mentioned in config_pred.yaml

## Deployment
Docker images for each component are built and pushed to Google Artifact Registry. These images are then used as Kubeflow components in the Vertex AI pipeline.
 - Deploying the Vertex AI pipeline (steps 6, 7, 8 above) - 
    - Prepare Your Pipeline Code (config_pipeline.py): Ensure your pipeline code is ready and you have defined your components and pipeline using the Kubeflow Pipelines SDK.
    - Compile the Pipeline (ml_pipeline.json): Compile your pipeline into a JSON format that Vertex AI can understand.
    - Upload the Pipeline Template to GCS: Store your compiled pipeline JSON on Google Cloud Storage (GCS).
    - Create and Run the Pipeline Job (run_pipeline.py): Use the Vertex AI SDK to create and run the pipeline job.

## Requirements
- Google Cloud SDK
- Vertex AI
- Kubeflow Pipelines
- Docker

## License
This project is licensed under the MIT License.