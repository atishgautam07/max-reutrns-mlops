from google.cloud import aiplatform

PROJECT_ID = "project name"
BUCKET_NAME = "bucket name"
REGION = "region name"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"

aiplatform.init(project=PROJECT_ID, location=REGION)

pipeline_job = aiplatform.PipelineJob(
    display_name='ml_pipeline',
    template_path=PIPELINE_ROOT + 'config/ml_pipeline.json',
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        'config_path': "gs://bucket name/folder/config_pred.yaml",
        'bucket_name': "bucket name"
    },
    enable_caching=False
)
pipeline_job.run()