from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

from airflow.utils.dates import days_ago

try:
    from airflow.utils.dates import days_ago
    from airflow.utils.state import State
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
except Exception:
    # If provider not installed, we'll still provide a pure-Python fallback
    pass

from airflow import AirflowException

from airflow.utils import timezone

# Import utils (these functions will be available when running inside the Airflow container
# that has pyspark and boto3 installed)
from airflow.utils.log.logging_mixin import LoggingMixin

# We import by relative path at runtime; in this file we use names expected in repo
try:
    from airflow.utils.dates import days_ago  # noqa: F401
    from ..utils.pipeline_utils import upload_file_to_minio, push_data_to_silver_layer, process_trusted_to_refined
except Exception:
    # When this module is syntax-checked in CI the relative import may fail; keep definitions safe
    def upload_file_to_minio(local_path: str, bucket: str, key: str):
        raise RuntimeError("upload_file_to_minio not available in this environment")

    def push_data_to_silver_layer(file_path: str):
        raise RuntimeError("push_data_to_silver_layer not available in this environment")

    def process_trusted_to_refined():
        raise RuntimeError("process_trusted_to_refined not available in this environment")

# Path to local sample CSV in this repo (used by the orchestrator)
LOCAL_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'content', 'files', 'netflix_titles.csv')

default_args = {
    'owner': 'Fia',
    'start_date': days_ago(1)
}

with DAG(dag_id='orchestrator_bronze_silver_refined', default_args=default_args, schedule_interval=None, catchup=False) as dag:

    def upload_to_bronze():
        if not os.path.exists(LOCAL_CSV):
            raise AirflowException(f"Local CSV not found: {LOCAL_CSV}")

        key = 'movies/netflix_titles.csv'
        upload_file_to_minio(LOCAL_CSV, bucket='bronze', key=key)
        return f"s3a://bronze/{key}"

    def run_silver(**context):
        bronze_s3a = f"s3a://bronze/movies/netflix_titles.csv"
        push_data_to_silver_layer(bronze_s3a)

    def run_refined(**context):
        process_trusted_to_refined()

    t_upload = PythonOperator(task_id='upload_to_bronze', python_callable=upload_to_bronze)
    t_silver = PythonOperator(task_id='run_silver', python_callable=run_silver, provide_context=True)
    t_refined = PythonOperator(task_id='run_refined', python_callable=run_refined, provide_context=True)

    t_upload >> t_silver >> t_refined
  
