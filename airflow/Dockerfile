# =====================================================================
# Dockerfile: Airflow + Spark 3.5.1 + Delta Lake + Hadoop AWS (S3A)
# =====================================================================
FROM ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.9.3-python3.11}

# ---------------------------------------------------------------------
# Instala Java e dependências básicas
# ---------------------------------------------------------------------
USER root
RUN apt-get update && \
    apt-get install -y wget curl gcc python3-dev openjdk-17-jdk && \
    apt-get clean

# Define JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# ---------------------------------------------------------------------
# Instala o Apache Spark (3.5.1 + Hadoop 3)
# ---------------------------------------------------------------------
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt/ && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
# ---------------------------------------------------------------------
# Adiciona conectores Hadoop AWS (para acessar MinIO com s3a://)
# ---------------------------------------------------------------------
RUN mkdir -p /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -O /opt/spark/jars/hadoop-aws-3.3.4.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar -O /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar

# ---------------------------------------------------------------------
# Variáveis de ambiente do Spark / Hadoop
# ---------------------------------------------------------------------
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$SPARK_HOME/bin:$HADOOP_HOME/bin
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JAVA_TOOL_OPTIONS="-Djava.net.preferIPv4Stack=true"

# ---------------------------------------------------------------------
# Instala dependências Python (Airflow + PySpark + Delta Lake + Kafka)
# ---------------------------------------------------------------------
USER airflow

COPY ./requirements.txt . 
RUN pip install -r requirements.txt
# ---------------------------------------------------------------------
# Copia DAGs e plugins do projeto
# ---------------------------------------------------------------------
COPY ./dags /opt/airflow/dags
COPY ./plugins /opt/airflow/plugins
