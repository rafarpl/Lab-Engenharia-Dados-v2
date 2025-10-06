from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, length, desc, split
from delta import configure_spark_with_delta_pip  # ⬅️ Delta Lake

# --------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# --------------------------------------------------------------------
MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "cursolab"
MINIO_SECRET_KEY = "cursolab"

# Trusted/Silver e Refined/Gold em Delta Lake
TRUSTED_PATH = "s3a://trusted/movies/"     # ⬅️ Delta
REFINED_PATH = "s3a://refined/movies/"     # ⬅️ Delta (destino das views)

# --------------------------------------------------------------------
# FUNÇÃO DE PROCESSAMENTO (DELTA IN/OUT)
# --------------------------------------------------------------------
def process_trusted_to_refined():
    # SparkSession com Delta + S3A (MinIO)
    builder = (
        SparkSession.builder.appName("TrustedToRefinedAirflow_Delta")
        # JARs locais do Hadoop AWS + SDK (necessários para s3a://)
        .config(
            "spark.jars",
            "/opt/spark/jars/hadoop-aws-3.3.4.jar,"
            "/opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar"
        )
        # MinIO/S3A
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Delta Lake (extensão + catálogo)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    print("Lendo dados da camada Trusted (Delta Lake)...")
    # ⬇️ Leia a Trusted como Delta (NÃO parquet)
    df = spark.read.format("delta").load(TRUSTED_PATH)

    # ----------------------------------------------------------------
    # Visão 1 — Contagem de títulos por tipo (Filme ou Série)
    # ----------------------------------------------------------------
    v1 = df.groupBy("type").agg(count("*").alias("total"))
    (
        v1.write.format("delta")
        .mode("overwrite")                    # sobrescreve a visão mantendo time travel
        .option("overwriteSchema", "true")    # segura se o schema evoluir
        .save(f"{REFINED_PATH}v1_titles_by_type")
    )

    # ----------------------------------------------------------------
    # Visão 2 — Top 10 países com mais produções
    # ----------------------------------------------------------------
    v2 = (
        df.groupBy("country")
          .agg(count("*").alias("total"))
          .orderBy(desc("total"))
          .limit(10)
    )
    (
        v2.write.format("delta")
        .mode("overwrite").option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v2_top10_countries")
    )

    # ----------------------------------------------------------------
    # Visão 3 — Duração média por categoria (Movie vs TV Show)
    # ----------------------------------------------------------------
    df_duration = df.withColumn("duration_num", split(col("duration"), " ").getItem(0).cast("int"))
    v3 = df_duration.groupBy("type").agg(avg("duration_num").alias("avg_duration"))
    (
        v3.write.format("delta")
        .mode("overwrite").option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v3_avg_duration")
    )

    # ----------------------------------------------------------------
    # Visão 4 — Quantidade de títulos por ano de lançamento
    # ----------------------------------------------------------------
    v4 = (
        df.filter(col("release_year").isNotNull())
          .groupBy("release_year")
          .agg(count("*").alias("total"))
          .orderBy("release_year")
    )
    (
        v4.write.format("delta")
        .mode("overwrite").option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v4_titles_by_year")
    )

    # ----------------------------------------------------------------
    # Visão 5 — Títulos com descrições mais longas
    # ----------------------------------------------------------------
    v5 = (
        df.withColumn("desc_length", length(col("description")))
          .select("title", "type", "desc_length")
          .orderBy(desc("desc_length"))
          .limit(10)
    )
    (
        v5.write.format("delta")
        .mode("overwrite").option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v5_longest_descriptions")
    )

    print("✅ Todas as visões foram processadas e salvas em Delta na camada Refined (Gold).")
    spark.stop()

# --------------------------------------------------------------------
# DAG AIRFLOW
# --------------------------------------------------------------------
default_args = {
    "owner": "Fia",
    "start_date": datetime(2025, 10, 4),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="trusted_to_refined_pyspark_delta",
    default_args=default_args,
    schedule_interval="*/10 * * * *",  # a cada 10 minutos (ajustei o cron)
    catchup=False,
    tags=["pyspark", "airflow", "refined", "delta", "lakehouse"],
) as dag:

    start = PythonOperator(
        task_id="start",
        python_callable=lambda: print("Iniciando processamento Trusted → Refined (Delta)"),
    )

    process = PythonOperator(
        task_id="process_trusted_to_refined",
        python_callable=process_trusted_to_refined,
    )

    end = PythonOperator(
        task_id="end",
        python_callable=lambda: print("🏁 Pipeline Trusted → Refined finalizado com sucesso"),
    )

    start >> process >> end
