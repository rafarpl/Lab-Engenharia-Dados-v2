# Lab

## Disclaimer
> **As configurações dos Laboratórios é puramente para fins de desenvolvimento local e estudos**


## Pré-requisitos?
* Docker
* Docker-Compose


```bash
docker compose up -d airflow-webserver  airflow-scheduler airflow-triggerer airflow-init airflow-cli
```

> http://localhost:8080/

* Username: airflow
* Password: airflow

## Orquestrador: Bronze → Silver → Refined

Um DAG chamado `orchestrator_bronze_silver_refined` foi adicionado em `airflow/dags/`.

O que ele faz:
- Faz upload do arquivo local `content/files/netflix_titles.csv` para o bucket `bronze` no MinIO.
- Executa uma tarefa que lê o CSV da camada Bronze e popula a tabela Delta na camada Trusted/Silver.
- Executa a transformação que gera as visões Delta na camada Refined.

Como testar (smoke test):
1. Levante o ambiente com `docker compose up -d` (veja `../docker-compose.yaml`).
2. Acesse a UI do Airflow em `http://localhost:8080` e confirme que o DAG `orchestrator_bronze_silver_refined` aparece.
3. Trigger manualmente o DAG (ou usar `airflow dags trigger orchestrator_bronze_silver_refined` via `airflow-cli` service).
4. Verifique os logs das tasks: `upload_to_bronze`, `run_silver`, `run_refined`.

Notas:
- O Spark e o Delta são usados dentro das tasks; certifique-se de que o serviço `spark` e o container `minio` estejam up caso execute as tasks que usam Spark.
- Os pacotes Python (boto3, pyspark, delta-spark) devem estar disponíveis no ambiente onde as tasks são executadas. Em setups com o `docker-compose` do repositório, você pode adicionar essas dependências via `_PIP_ADDITIONAL_REQUIREMENTS` nas variáveis de ambiente ou estendendo a imagem do Airflow.

Instalando dependências Python no Airflow (recomendações)
1) Arquivo `.env` (já incluído) contém a variável `_PIP_ADDITIONAL_REQUIREMENTS` com as libs:
	```
	_PIP_ADDITIONAL_REQUIREMENTS=boto3 pyspark delta-spark
	```
	O `docker-compose.yaml` do projeto já lê essa variável ao iniciar os containers e instalará essas dependências no startup do container do Airflow.

2) Para forçar a instalação (recomendado após adicionar/alterar `.env`):
	Pare os containers e remova volumes temporários usados pelo `airflow-init`, então suba novamente com rebuild:
	```bash
	docker compose down
	docker compose up -d --build airflow-init airflow-webserver airflow-scheduler
	```
	Observe os logs do serviço `airflow-init` para ver o processo de instalação de dependências.

3) Nota: instalar `pyspark` e `delta-spark` pode levar alguns minutos no primeiro startup, devido ao download de dependências.

