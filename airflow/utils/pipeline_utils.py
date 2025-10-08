import os
import urllib.parse
import boto3
from botocore.client import Config
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, upper, count, avg, length, desc, split
from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip

# MinIO / S3 configuration (kept same as existing DAGs)
MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "cursolab"
MINIO_SECRET_KEY = "cursolab"


def upload_file_to_minio(local_path: str, bucket: str, key: str):
    """Upload a local file to MinIO S3-compatible storage.

    local_path: local filesystem path to file
    bucket: target bucket name (e.g. 'bronze')
    key: object key inside bucket (e.g. 'movies/netflix_titles.csv')
    """
    client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )

    # Ensure bucket exists (quietly ignore if already exists)
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)

    client.upload_file(local_path, bucket, key)


SILVER_PATH = "s3a://trusted/movies/"
REFINED_PATH = "s3a://refined/movies/"


def _create_spark_session(app_name: str = "pipeline"):
    builder = (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars",
            "/opt/spark/jars/hadoop-aws-3.3.4.jar,/opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar",
        )
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


def push_data_to_silver_layer(file_path: str):
    """Read CSV at file_path (local or s3a://) with Spark, clean and upsert into Delta Silver path."""
    spark = _create_spark_session("BronzeToSilver")

    df = spark.read.option("header", True).csv(file_path)

    df_clean = (
        df.dropDuplicates(["show_id"]).withColumn("title", trim(col("title")))
        .withColumn("type", upper(col("type")))
        .withColumn("country", upper(col("country")))
        .na.drop(subset=["title", "type", "country"])
    )

    try:
        delta_table = DeltaTable.forPath(spark, SILVER_PATH)
        (
            delta_table.alias("t")
            .merge(df_clean.alias("s"), "t.show_id = s.show_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    except Exception:
        df_clean.write.format("delta").mode("overwrite").save(SILVER_PATH)

    spark.stop()


def process_trusted_to_refined():
    """Read Trusted (Silver) Delta table and write a set of analytical views to Refined (Delta)."""
    spark = _create_spark_session("TrustedToRefined")

    df = spark.read.format("delta").load(SILVER_PATH)

    # v1 — count titles by type
    v1 = df.groupBy("type").agg(count("*").alias("total"))
    (
        v1.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v1_titles_by_type")
    )

    # v2 — top10 countries
    v2 = df.groupBy("country").agg(count("*").alias("total")).orderBy(desc("total")).limit(10)
    (
        v2.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v2_top10_countries")
    )

    # v3 — average duration
    df_duration = df.withColumn("duration_num", split(col("duration"), " ").getItem(0).cast("int"))
    v3 = df_duration.groupBy("type").agg(avg("duration_num").alias("avg_duration"))
    (
        v3.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v3_avg_duration")
    )

    # v4 — titles by year
    v4 = df.filter(col("release_year").isNotNull()).groupBy("release_year").agg(count("*").alias("total")).orderBy("release_year")
    (
        v4.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v4_titles_by_year")
    )

    # v5 — longest descriptions
    v5 = df.withColumn("desc_length", length(col("description"))).select("title", "type", "desc_length").orderBy(desc("desc_length")).limit(10)
    (
        v5.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{REFINED_PATH}v5_longest_descriptions")
    )

    spark.stop()
