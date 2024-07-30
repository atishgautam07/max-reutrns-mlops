import datetime
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from sqlalchemy import create_engine
from google.cloud import storage
from config_entity import (ModelMonitoringConfig)
import config

class ModelMonitoring:
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        self.storage_client = storage.Client()

        self.X_all = pd.read_parquet(self.config.data_path + '/X_all.parquet')
        self.y_all = pd.read_parquet(self.config.data_path + '/y_all.parquet')
        self.df_full = pd.read_parquet(self.config.data_path + '/dfOrigData.parquet')
        # self.X_train_valid = pd.read_parquet(self.config.data_path + '/X_train_valid.parquet')
        # self.y_train_valid = pd.read_parquet(self.config.data_path + '/y_train_valid.parquet')
        # self.X_test = pd.read_parquet(self.config.data_path + '/X_test.parquet')
        # self.y_test = pd.read_parquet(self.config.data_path + '/y_test.parquet')
        
        self.df_full['Date'] = pd.to_datetime(self.df_full['Date']).dt.date
        self.begin = self.df_full['Date'].max() - datetime.timedelta(31)
        # self.begin = datetime.date(2024, 6, 1)
        print (self.begin)


    def upload_to_gcs(self, gcs_path, local_path):
        gcs_path = '/'.join(gcs_path.split('/')[3:])
        client = self.storage_client
        bucket = client.get_bucket(self.config.bucketName)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f'File {local_path} uploaded to gs://{self.config.bucketName}/{gcs_path}')

    def download_from_gcs(self, gcs_path, local_path):
        gcs_path = '/'.join(gcs_path.split('/')[3:])
        client = storage.Client()
        bucket = client.get_bucket(self.config.bucketName)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f'File gs://{self.config.bucketName}/{gcs_path} downloaded to {local_path}') 

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
        # print (self.df_full.columns.to_list())
        self.df_full = self.df_full[['Date','Adj Close',  'ticker_type', 'Ticker', 'Year', 'Month', 'Weekday','growth_1d', 'growth_3d', 'growth_7d', 'growth_30d', 'growth_90d', 'growth_365d', 
                                     'SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative', 'volatility', 'DGS1', 'DGS5', 'Quarter', 'gdppot_us_yoy', 'gdppot_us_qoq',
                                     'FEDFUNDS', 'cpi_core_yoy', 'cpi_core_mom','DGS10', 'GVZCLS', 'growth_snp500_1d', 'ln_volume',
                                     'split','growth_future_5d', 'is_positive_growth_5d_future', 'pred_xgp_rf_best']]
        self.df_full['target'] = 1 * self.df_full['is_positive_growth_5d_future']
        self.df_full['prediction'] = 1 * (self.df_full['pred_xgp_rf_best'] >= 0.5)
        self.df_full['Date'] = pd.to_datetime(self.df_full['Date']).dt.date
        print (self.df_full['Date'].max())

    def run_monitoring(self, i, reference_data, current_data):

        print (current_data.shape)
        # Initialize Evidently report
        # report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report = Report(metrics = [
                            ColumnDriftMetric(column_name='target'),
                            ColumnDriftMetric(column_name='prediction'),
                            DatasetDriftMetric(),
                            DatasetMissingValuesMetric()
                        ])
        report.run(reference_data=reference_data, current_data=current_data)

        # Save Evidently report
        report_json_path = "evidently_report.json"
        report_html_path = "evidently_dashboard.html"
        
        report.save_json(report_json_path)
        report.save_html(report_html_path)
        
        # Upload reports to GCS
        self.upload_to_gcs(gcs_path=self.config.root_dir+"/evidently_report.json", local_path=report_json_path)
        self.upload_to_gcs(gcs_path=self.config.root_dir+"/evidently_dashboard.html", local_path=report_html_path) 
        
        # Log to MLflow
        with open(report_json_path) as f:
            mlflow.log_text(f.read(), "evidently_report.json")

        mlflow.log_artifact(report_html_path)

        # Load the JSON report
        with open(report_json_path) as f:
            report_data = json.load(f)

        result = report_data#.as_dict()

        target_drift = result['metrics'][0]['result']['drift_score']
        prediction_drift = result['metrics'][1]['result']['drift_score']
        share_of_drifted_columns = result['metrics'][2]['result']['share_of_drifted_columns']
        share_missing_values = result['metrics'][3]['result']['current']['share_of_missing_values']

        return {"Date": self.begin + datetime.timedelta(i),
                "target_drift": target_drift,
                "prediction_drift": prediction_drift, 
                "share_of_drifted_columns":share_of_drifted_columns, 
                "share_missing_values": share_missing_values}

    def write_metrics(self):

        # Load the data
        ref_data = self.df_full[self.df_full.split.isin(['train','validation'])].copy(deep=True) #pd.concat([self.X_train_valid, self.y_train_valid])
        curr_data = self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)  #pd.concat([self.X_test, self.y_test])

        print ("Updating sql table with drift metrics.")
        for i in range(1, 30):
            # drift_df = pd.DataFrame([self.run_monitoring(i, ref_data, curr_data[(curr_data.Date >= (self.begin + datetime.timedelta(i))) &
            #                                                                    (curr_data.Date < (self.begin + datetime.timedelta(i + 1)))])])
            drift_df = pd.DataFrame([self.run_monitoring(i, ref_data, curr_data[(curr_data.Date >= (self.begin + datetime.timedelta(i - 30))) &
                                                                                (curr_data.Date < (self.begin + datetime.timedelta(i + 1)))])])
            
            
            engine = create_engine(self.config.DB_URI)
            dftmp = pd.read_sql(f"select * from {self.config.TABLE_NAME};", engine)
            dftmp = pd.concat([dftmp, drift_df])
            #send data back to sql
            dftmp.to_sql(f"{self.config.TABLE_NAME}", engine, if_exists='replace', index=False)