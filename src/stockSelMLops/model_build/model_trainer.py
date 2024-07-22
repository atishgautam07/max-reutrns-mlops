import os
import pandas as pd
import mlflow
import numpy as np

# ML models and utils
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from google.cloud import storage

from config_entity import (ModelTrainerConfig)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.storage_client = storage.Client()
    
        self.X_train_valid = pd.read_parquet(self.config.data_path + '/X_train_valid.parquet')
        self.y_train_valid = pd.read_parquet(self.config.data_path + '/y_train_valid.parquet').values.ravel()
    
    
    def train_rf(self):

        # def run_optimization(data_path: str, num_trials: int):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.hpo_exp_rf)
        

        def objective_rf(params):
            with mlflow.start_run():
                mlflow.set_tag("model", "randomforest")
                mlflow.log_params(params)

                model = RandomForestClassifier(**params, random_state=42)
                model.fit(self.X_train_valid.to_numpy(), self.y_train_valid)
                score = cross_val_score(model, self.X_train_valid, self.y_train_valid, cv=self.config.cv, scoring='precision').mean()
            
                mlflow.log_metric("precision", score)
                mlflow.sklearn.log_model(model, artifact_path="model")
            return {'loss': -score, 'status': STATUS_OK}
    
        # Hyperparameter space for RandomForest
        rf_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 5)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 4)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 2)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1))
        }
    
    
        # Running hyperparameter optimization
        print ("Running hyperparameter optimization for RF.")
        rf_trials = Trials()
        best_rf = fmin(fn=objective_rf, space=rf_space, algo=tpe.suggest, max_evals=self.config.num_trials, trials=rf_trials, rstate=np.random.default_rng(42))
    
        print (best_rf)
        

    def train_xgb(self):

        # def run_optimization(data_path: str, num_trials: int):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.hpo_exp_xgb)
        
        def objective_xgb(params):
            with mlflow.start_run():
                mlflow.set_tag("model", "xgboost")
                mlflow.log_params(params)

                model = XGBClassifier(**params, random_state=42,
                                    #   use_label_encoder=False, 
                                      eval_metric='mlogloss')
                
                model.fit(self.X_train_valid, self.y_train_valid)
                score = cross_val_score(model, self.X_train_valid, self.y_train_valid, cv=self.config.cv, scoring='precision').mean()
            
                mlflow.log_metric("precision", score)
                mlflow.xgboost.log_model(model, artifact_path="model")
            return {'loss': -score, 'status': STATUS_OK}
    
        # Hyperparameter space for XGBoost
        xgb_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 5)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 4)),
            'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }
    
        # Running hyperparameter optimization
        print ("Running hyperparameter optimization for XGBoost.")
        xgb_trials = Trials()
        best_xgb = fmin(fn=objective_xgb, space=xgb_space, algo=tpe.suggest, max_evals=self.config.num_trials, trials=xgb_trials, rstate=np.random.default_rng(42))

        print (best_xgb)    
