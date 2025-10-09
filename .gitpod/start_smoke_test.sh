#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
echo "Root dir: $ROOT_DIR"

function wait_for() {
  local hostport=$1
  local timeout=${2:-60}
  local start=$(date +%s)
  while true; do
    if nc -z $(echo $hostport | sed 's/:/ /'); then
      echo "$hostport is reachable"
      return 0
    fi
    now=$(date +%s)
    if (( now - start > timeout )); then
      echo "Timeout waiting for $hostport" >&2
      return 1
    fi
    sleep 2
  done
}

echo "Waiting for MinIO (9000) and Airflow webserver (8080)..."
wait_for 127.0.0.1:9000 90
wait_for 127.0.0.1:8080 120

echo "Triggering DAG orchestrator_bronze_silver_refined via airflow-cli container..."
docker compose run --rm airflow-cli airflow dags trigger orchestrator_bronze_silver_refined || true

echo "Sleeping 10s to allow tasks to start..."
sleep 10

echo "Collecting recent logs from airflow-webserver and minio (tail)..."
docker compose logs --tail 200 airflow-webserver | sed -n '1,200p'
docker compose logs --tail 200 minio | sed -n '1,200p'

echo "Smoke test triggered; abra o Airflow UI em http://localhost:8080 para acompanhar o DAG."
