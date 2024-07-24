import os
import mlflow
import pandas as pd
import pickle
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import  precision_score
from google.cloud import storage
from config_entity import (ModelEvaluationConfig)
import xgboost as xgb



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.storage_client = storage.Client()

        self.X_train_valid = pd.read_parquet(self.config.data_path + '/X_train_valid.parquet')
        self.y_train_valid = pd.read_parquet(self.config.data_path + '/y_train_valid.parquet').values.ravel()
        self.X_test = pd.read_parquet(self.config.data_path + '/X_test.parquet')
        self.y_test = pd.read_parquet(self.config.data_path + '/y_test.parquet').values.ravel()
        self.X_valid = pd.read_parquet(self.config.data_path + '/X_valid.parquet')
        self.y_valid = pd.read_parquet(self.config.data_path + '/y_valid.parquet').values.ravel()

    def load_pickle(self, filename):
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)

    def run_register_model_rf(self):

        mlflow.set_tracking_uri(self.config.ml_uri)
        mlflow.set_experiment(self.config.exp_name)
        mlflow.sklearn.autolog()
        client = MlflowClient(tracking_uri=self.config.ml_uri)

        # Retrieve the top_n model runs and log the models
        print("Retrieve the top_n model runs and log the models.")
        experiment = client.get_experiment_by_name(self.config.hpo_exp_rf)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.precision DESC"]
        )
        print(len(runs))
        print("logging top_n models wiht test metrics.")
        
        for run in runs:
            print((str(run.info.run_id), str(run.data.metrics), str(run.data.params)))
            modelPath = client.download_artifacts(run_id=run.info.run_id, path="model")
            pipeLine = self.load_pickle(os.path.join(modelPath, "model.pkl"))
            

            with mlflow.start_run():

                mlflow.set_tag("model", "rf_topN_models")
                mlflow.log_params(run.data.params)
                
                pipeLine.fit(self.X_train_valid.to_numpy(), self.y_train_valid)
                print("Evaluate model on the validation and test sets")
                val_score = precision_score(self.y_valid, pipeLine.predict(self.X_valid.to_numpy()))
                mlflow.log_metric("val_score", val_score)
                test_score = precision_score(self.y_test, pipeLine.predict(self.X_test.to_numpy()))
                mlflow.log_metric("test_score", test_score)
                mlflow.sklearn.log_model(pipeLine, artifact_path="model")

        print("Selecting the model with the lowest test score")
        experiment = client.get_experiment_by_name(self.config.exp_name)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            filter_string='tags.model="rf_topN_models"',
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.test_score DESC"]
        )[0]

        # Register the best model
        print("Registering the best RF model")
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="best-model-rf")


    def run_register_model_xgb(self):

        mlflow.set_tracking_uri(self.config.ml_uri)
        mlflow.set_experiment(self.config.exp_name)
        mlflow.xgboost.autolog()
        client = MlflowClient(tracking_uri=self.config.ml_uri)
        
        # Retrieve the top_n model runs and log the models
        print("Retrieve the top_n model runs and log the models.")
        experiment = client.get_experiment_by_name(self.config.hpo_exp_xgb)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.precision DESC"]
        )
        print(len(runs))
        print("logging top_n models wiht test metrics.")
        
        for run in runs:
            print((str(run.info.run_id), str(run.data.metrics), str(run.data.params)))
            modelPath = client.download_artifacts(run_id=run.info.run_id, path="model")
            pipeLine = mlflow.xgboost.load_model(modelPath)#os.path.join(modelPath, "model.xgb"))

            with mlflow.start_run():

                mlflow.set_tag("model", "xgb_topN_models")
                mlflow.log_params(run.data.params)
                
                
                print("Evaluate model on the validation and test sets")
                dvalid = xgb.DMatrix(self.X_valid, label=self.y_valid)
                y_predVal = pipeLine.predict(dvalid)
                y_pred_binary = (y_predVal > 0.5).astype(int)
                precision_val = precision_score(self.y_valid, y_pred_binary)
                mlflow.log_metric("val_score", precision_val)

                dtest = xgb.DMatrix(self.X_test, label=self.y_test)
                y_predTest = pipeLine.predict(dtest)
                y_pred_binary = (y_predTest > 0.5).astype(int)
                precision_test = precision_score(self.y_test, y_pred_binary)
                mlflow.log_metric("test_score", precision_test)
                
                mlflow.xgboost.log_model(pipeLine, artifact_path="model")

                # val_score = precision_score(self.y_valid, pipeLine.predict(self.X_valid.to_numpy()))
                # mlflow.log_metric("val_score", val_score)
                # test_score = precision_score(self.y_test, pipeLine.predict(self.X_test.to_numpy()))
                # mlflow.log_metric("test_score", test_score)
                # mlflow.xgboost.log_model(pipeLine, artifact_path="model")

        print("Selecting the model with the lowest test score")
        experiment = client.get_experiment_by_name(self.config.exp_name)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            filter_string='tags.model="xgb_topN_models"',
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.test_score DESC"]
        )[0]

        # Register the best model
        print ("Registering the best XGB model")
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="best-model-xgb")
