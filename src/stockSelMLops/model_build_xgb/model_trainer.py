import pandas as pd
import mlflow
import numpy as np

# ML models and utils
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
# from google.cloud import storage

from config_entity import (ModelTrainerConfig)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        # self.storage_client = storage.Client()
    
        self.X_train_valid = pd.read_parquet(self.config.data_path + '/X_train_valid.parquet')
        self.y_train_valid = pd.read_parquet(self.config.data_path + '/y_train_valid.parquet')#.values.ravel()
        # self.X_valid = pd.read_parquet(self.config.data_path + '/X_valid.parquet')
        # self.y_valid = pd.read_parquet(self.config.data_path + '/y_valid.parquet').values.ravel()
    

    def train_xgb(self):

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.hpo_exp_xgb)
        
        def objective_xgb(params):
            with mlflow.start_run():
                mlflow.set_tag("model", "xgboost")
                mlflow.log_params(params)

                # Extract and use only the necessary parameters
                params = {
                            'max_depth': int(params['max_depth']),
                            'subsample': params['subsample'],
                            'colsample_bytree': params['colsample_bytree'],
                            'eta': params['eta'],
                            'eval_metric': 'logloss',
                            'objective': 'binary:logistic',
                            'learning_rate': params['learning_rate']  # Include learning rate
                        }
                
                skf = TimeSeriesSplit(n_splits=self.config.cv)
                precision_scores = []

                for train_index, valid_index in skf.split(self.X_train_valid, self.y_train_valid):
                    X_train, X_valid = self.X_train_valid.iloc[train_index], self.X_train_valid.iloc[valid_index]
                    y_train, y_valid = self.y_train_valid.iloc[train_index], self.y_train_valid.iloc[valid_index]

                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dvalid = xgb.DMatrix(X_valid, label=y_valid)

                    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
                    model = xgb.train(params, dtrain, num_boost_round=1000,
                                    evals=watchlist,
                                    early_stopping_rounds=50,
                                    verbose_eval=False)

                    y_pred = model.predict(dvalid)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    precision = precision_score(y_valid, y_pred_binary)
                    precision_scores.append(precision)

                score = np.mean(precision_scores)
                mlflow.log_metric("precision", score)
                mlflow.xgboost.log_model(model, artifact_path="model")
                return {'loss': -score, 'status': STATUS_OK}
        
        # Define the search space
        xgb_space = {
                    'max_depth': hp.quniform('max_depth', 3, 20, 3),
                    'subsample': hp.uniform('subsample', 0.7, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
                    'eta': hp.loguniform('eta', -3, 0),
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),  # Learning rate
                    # 'num_boost_round': hp.quniform('num_boost_round', 75, 1000, 100)  # Number of boosting rounds
                }

        # Running hyperparameter optimization
        print ("Running hyperparameter optimization for XGBoost.")
        xgb_trials = Trials()
        best_xgb = fmin(fn=objective_xgb, space=xgb_space,
                        algo=tpe.suggest, max_evals=self.config.num_trials,
                        trials=xgb_trials, rstate=np.random.default_rng(42))

        print (best_xgb)    



            


    # def train_xgb(self):

    #     # def run_optimization(data_path: str, num_trials: int):
    #     mlflow.set_tracking_uri(self.config.mlflow_uri)
    #     mlflow.set_experiment(self.config.hpo_exp_xgb)
        
    #     def objective_xgb(params):
    #         with mlflow.start_run():
    #             mlflow.set_tag("model", "xgboost")
    #             mlflow.log_params(params)

    #             model = XGBClassifier(**params, random_state=42, n_jobs=3,
    #                                 #   use_label_encoder=False, 
    #                                   eval_metric='mlogloss')
                
    #             model.fit(self.X_train_valid, self.y_train_valid)
    #             score = cross_val_score(model, self.X_train_valid, self.y_train_valid, cv=self.config.cv, scoring='precision').mean()
            
    #             mlflow.log_metric("precision", score)
    #             mlflow.xgboost.log_model(model, artifact_path="model")
    #         return {'loss': -score, 'status': STATUS_OK}
    
    #     # Hyperparameter space for XGBoost
    #     xgb_space = {
    #         'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 5)),
    #         'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 4)),
    #         'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
    #         'subsample': hp.uniform('subsample', 0.6, 1.0),
    #         'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    #     }
    
    #     # Running hyperparameter optimization
    #     print ("Running hyperparameter optimization for XGBoost.")
    #     xgb_trials = Trials()
    #     best_xgb = fmin(fn=objective_xgb, space=xgb_space, algo=tpe.suggest, max_evals=self.config.num_trials, trials=xgb_trials, rstate=np.random.default_rng(42))

    #     print (best_xgb)    
