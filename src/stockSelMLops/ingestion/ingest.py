from configuration import ConfigurationManager
from data_ingestion import DataIngestion

import argparse
import pandas as pd
pd.options.mode.chained_assignment = None


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(bucket_name, gcs_file_path)
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        if data_ingestion_config.fetchRepo:
            # Fetch All 3 datasets for all dates from APIs
            data_ingestion.fetch()
            # save data to a local dir
            data_ingestion.persist()
        else:
            # OR Load from disk
            data_ingestion.load()  



parser = argparse.ArgumentParser(description='Fetch data from a URL.')
parser.add_argument('--bucket', type=str, required=True, help='Bucket name')
parser.add_argument('--config', type=str, required=True, help='The path to the configuration YAML file')
args = parser.parse_args()

bucket_name = args.bucket
print (bucket_name)
gcs_file_path = args.config
print(gcs_file_path)

try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e


