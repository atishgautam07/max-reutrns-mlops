import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import storage
from config_entity import (ModelPredictionConfig)

class PredictionPipeline:
    def __init__(self, config: ModelPredictionConfig):
        self.config = config
        self.storage_client = storage.Client()

        self.X_all = pd.read_parquet(self.config.data_path + '/X_all.parquet')
        self.y_all = pd.read_parquet(self.config.data_path + '/y_all.parquet')
        self.df_full = pd.read_parquet(self.config.data_path + '/dfOrigData.parquet')


    def load(self):
        """Load files from the local directory"""
        
        mlflow.set_tracking_uri(self.config.ml_uri)
        mlflow.set_experiment(self.config.exp_name)
        
        # client = MlflowClient(tracking_uri=self.ml_uri)
        model_version = "latest"

        model_uri_rf = f"models:/{self.config.model_name_rf}/{model_version}"
        model_uri_xgb = f"models:/{self.config.model_name_xgb}/{model_version}"
        
        self.best_rf_model = mlflow.sklearn.load_model(model_uri_rf)
        self.best_xgb_model = mlflow.xgboost.load_model(model_uri_xgb)

    def predict(self):
        print('Making inference')

        y_pred_all_rf = self.best_rf_model.predict_proba(self.X_all)
        y_pred_all_class1_rf = [k[1] for k in y_pred_all_rf] #list of predictions for class "1"
        y_pred_all_class1_array_rf = np.array(y_pred_all_class1_rf) # (Numpy Array) np.array of predictions for class "1" , converted from a list

        dall = xgb.DMatrix(self.X_all, label=self.y_all)
        y_pred_all_class1_xgb = self.best_xgb_model.predict(dall)
        # y_pred_binary = (y_predTest > 0.5).astype(int)
        # y_pred_all_xgb = self.best_xgb_model.predict_proba(dall) #self.X_all)
        # y_pred_all_class1_xgb = [k[1] for k in y_pred_all_xgb] #list of predictions for class "1"
        y_pred_all_class1_array_xgb = np.array(y_pred_all_class1_xgb) # (Numpy Array) np.array of predictions for class "1" , converted from a list

        self.df_full[self.config.prediction_name + "_rf"] = y_pred_all_class1_array_rf
        self.df_full[self.config.prediction_name + "_xgb"] = y_pred_all_class1_array_xgb

        self.df_full[self.config.prediction_name] = (self.df_full[self.config.prediction_name + "_rf"] + self.df_full[self.config.prediction_name + "_xgb"]) / 2.0

        # define rank of the prediction
        self.df_full[f"{self.config.prediction_name}_rank"] = self.df_full.groupby("Date")[self.config.prediction_name].rank(method="first", ascending=False)

        self.df_full[['Date','Ticker','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                          'growth_future_5d', 'is_positive_growth_5d_future','split', 
                                          'pred_xgp_rf_best', 'pred_xgp_rf_best_rank']].to_csv(self.config.root_dir + "/df_pred_forSim.csv")

        COLUMNS = ['Ticker', 'Adj Close','Date',self.config.prediction_name, self.config.prediction_name+'_rank']
        result_df = self.df_full[(self.df_full[f'{self.config.prediction_name}_rank']<=10) & 
                                    (self.df_full['Date'] == self.df_full['Date'].max())].sort_values(by=self.config.prediction_name)[COLUMNS]
        result_df.to_csv(self.config.root_dir + "/top10Stocks.csv")
        