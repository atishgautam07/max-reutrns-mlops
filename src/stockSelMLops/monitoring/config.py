import os

## "postgresql://postgres:mlflowsma@10.2.80.3:5432/evidai_v1"

DB_HOST = os.getenv("DB_HOST", "10.2.80.3")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "evidai_v1")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlflowsma")

DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
TABLE_NAME = os.getenv("TABLE_NAME", "evidently_metrics")
BUCKET_NAME = os.getenv("BUCKET_NAME", "sma-proj-bucket")
# REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "reference_data.csv")
# CURRENT_DATA_PATH = os.getenv("CURRENT_DATA_PATH", "current_data.csv")
# OUTPUT_PATH = os.getenv("OUTPUT_PATH", "monitoring_output")
