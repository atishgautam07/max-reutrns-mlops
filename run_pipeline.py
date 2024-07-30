from google.cloud import aiplatform

PROJECT_ID = "sma-mlops"
BUCKET_NAME = "sma-proj-bucket"
REGION = "asia-south1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"

aiplatform.init(project=PROJECT_ID, location=REGION)

pipeline_job = aiplatform.PipelineJob(
    display_name='ml_pipeline',
    template_path=PIPELINE_ROOT + 'config/ml_pipeline.json',
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        'config_path': "gs://sma-proj-bucket/pipeline_root/config_pred.yaml",
        'bucket_name': "sma-proj-bucket"
    },
    enable_caching=False
)
pipeline_job.run()