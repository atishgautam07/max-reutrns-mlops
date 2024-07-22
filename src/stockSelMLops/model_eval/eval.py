from configuration import ConfigurationManager
# from data_ingestion import DataIngestion
# from data_transformation import DataTransformation
# from model_trainer import ModelTrainer
from model_evaluation import ModelEvaluation

import argparse
import pandas as pd
pd.options.mode.chained_assignment = None


# class DataIngestionTrainingPipeline:
#     def __init__(self):
#         pass

#     def main(self):
#         config = ConfigurationManager(bucket_name, gcs_file_path)
#         data_ingestion_config = config.get_data_ingestion_config()
#         data_ingestion = DataIngestion(config=data_ingestion_config)
#         if data_ingestion_config.fetchRepo:
#             # Fetch All 3 datasets for all dates from APIs
#             data_ingestion.fetch()
#             # save data to a local dir
#             data_ingestion.persist()
#         else:
#             # OR Load from disk
#             data_ingestion.load()  


# class DataTransformationTrainingPipeline:
#     def __init__(self):
#         pass

#     def main(self):
#         config = ConfigurationManager(bucket_name, gcs_file_path)
#         data_transformation_config = config.get_data_transformation_config()
#         data_transformation = DataTransformation(config=data_transformation_config)
#         if data_transformation_config.transfData:
#             data_transformation.transform()
#             data_transformation.persist()
#             data_transformation.prepare_dataframe()
#         else:
#             data_transformation.load()
#             data_transformation.prepare_dataframe()


# class ModelTrainerTrainingPipeline:
#     def __init__(self):
#         pass

#     def main(self):
#         config = ConfigurationManager(bucket_name, gcs_file_path)
#         model_trainer_config = config.get_model_trainer_config()
#         model_trainer = ModelTrainer(config=model_trainer_config)
#         if model_trainer_config.trainModel:
#             model_trainer.train_rf()
#             model_trainer.train_xgb()
#         else:
#             pass


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(bucket_name, gcs_file_path)
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        if model_evaluation_config.trainModel:
            model_evaluation.run_register_model_rf()
            model_evaluation.run_register_model_xgb()
        else:
            pass


parser = argparse.ArgumentParser(description='Fetch data from a URL.')
parser.add_argument('--bucket', type=str, required=True, help='Bucket name')
parser.add_argument('--config', type=str, required=True, help='The path to the configuration YAML file')
args = parser.parse_args()

bucket_name = args.bucket
print (bucket_name)
gcs_file_path = args.config
print(gcs_file_path)


# STAGE_NAME = "Data Ingestion stage"
# try:
#    print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         print(e)
#         raise e

# STAGE_NAME = "Data Transformation stage"
# try:
#     print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_transformation = DataTransformationTrainingPipeline()
#     data_transformation.main()
#     print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     print(e)
#     raise e

# STAGE_NAME = "Model training stage"
# try:
#     print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     model_train = ModelTrainerTrainingPipeline()
#     model_train.main()
#     print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     print(e)
#     raise e

STAGE_NAME = "Model Evaluation stage"
try:
    print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_eval = ModelEvaluationTrainingPipeline()
    model_eval.main()
    print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    print(e)
    raise e
