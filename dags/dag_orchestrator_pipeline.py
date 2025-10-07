from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

from airflow.utils.dates import days_ago

# Local imports from utils
from airflow.utils import timezone

try:
    from airflow.utils.dates import days_ago
    from airflow.utils.state import State
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
except Exception:
    # If provider not installed, we'll still provide a pure-Python fallback
    pass

from airflow import AirflowException

from ..utils.pipeline_utils import upload_file_to_minio, push_data_to_silver_layer, process_trusted_to_refined

# Path to local sample CSV in this repo
LOCAL_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'content', 'files', 'netflix_titles.csv')

default_args = {
    'owner': 'Fia',
    'start_date': days_ago(1)
}

with DAG(dag_id='orchestrator_bronze_silver_refined', default_args=default_args, schedule_interval=None, catchup=False) as dag:

    def upload_to_bronze():
        if not os.path.exists(LOCAL_CSV):
            raise AirflowException(f"Local CSV not found: {LOCAL_CSV}")

        # Key in bronze bucket
        key = 'movies/netflix_titles.csv'
        upload_file_to_minio(LOCAL_CSV, bucket='bronze', key=key)
        return f"s3a://bronze/{key}"

    def run_silver(**context):
        # The file is now in bronze; we'll read it from s3a path
        bronze_s3a = f"s3a://bronze/movies/netflix_titles.csv"
        push_data_to_silver_layer(bronze_s3a)

    def run_refined(**context):
        process_trusted_to_refined()

    t_upload = PythonOperator(task_id='upload_to_bronze', python_callable=upload_to_bronze)
    t_silver = PythonOperator(task_id='run_silver', python_callable=run_silver, provide_context=True)
    t_refined = PythonOperator(task_id='run_refined', python_callable=run_refined, provide_context=True)

    t_upload >> t_silver >> t_refined
