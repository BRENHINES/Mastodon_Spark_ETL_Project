import os, subprocess
from dagster import op, OpExecutionContext, Field, In, Nothing, Out
import requests
import json
import time

# from spark.streaming_to_postgres import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_PASSWORD, POSTGRES_USER

PROJECT_ROOT = "/opt/project"
SPARK_CONTAINER = "spark-mastodon-core-spark-master-1"
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
SENTIMENT_TRAIN_TABLE = os.getenv("SENTIMENT_TRAIN_TABLE", "sentiment140_raw")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB   = os.getenv("POSTGRES_DB", "toots")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


@op(
    config_schema={
        "duration_secs": Field(int, description="Durée max du stream en secondes", default_value=300),
        "max_toots": Field(int, description="Nombre max de toots avant arrêt", default_value=500),
        "extra_env": Field(dict, description="Env additionnelle facultative", default_value={})
    },
    tags={"module": "stream"},
    ins={"start": In(Nothing)},
    out=None
)
def run_stream_for_window(context: OpExecutionContext) -> None:
    duration = context.op_config.get("duration_secs", 300)
    max_toots = context.op_config.get("max_toots", 500)

    env = os.environ.copy()

    # Permettre d'injecter MASTODON_ACCESS_TOKEN ou overrides si besoin
    for k, v in context.op_config.get("extra_env", {}).items():
        env[str(k)] = str(v)

    cmd = [
        "python", "/opt/project/spark/streaming_to_postgres.py",
        "--duration-sec", str(duration),
        "--max-toots", str(max_toots),
    ]

    context.log.info(f"Launching stream: {' '.join(cmd)} (duration={duration}s, max_toots={max_toots})")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
    context.log.info("Stream finished successfully.")


@op(
    ins={"start": In(Nothing)},
    tags={"module": "batch"},
)
def run_batch_aggregations(context: OpExecutionContext) -> None:
    """
    Exécute le job Spark via le container spark-master
    """
    # Préparer les variables d'environnement pour le container Spark
    env_vars = []
    for key in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_TABLE",
                "SPARK_MASTER"]:
        value = os.getenv(key, "")
        if value:
            env_vars.extend(["-e", f"{key}={value}"])

    # Commande docker exec
    cmd = [
        "docker", "exec",
        *env_vars,
        "spark-mastodon-core-spark-master-1",  # ← Nom du container depuis votre docker-compose
        "/opt/bitnami/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "--jars", "/opt/spark/jars/postgresql-42.7.4.jar",
        "/opt/project/spark/batch_aggregations.py"
    ]

    context.log.info(f"Running batch via spark-master: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.stdout:
        context.log.info(f"📄 STDOUT:\n{result.stdout}")
    if result.stderr:
        context.log.error(f"❌ STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise Exception(
            f"Spark job failed with exit code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )

    context.log.info("✅ Batch finished successfully.")


@op(out=Out(str), tags={"module": "ml"})
def train_sentiment_model(context: OpExecutionContext) -> str:
    """Entraîne un modèle de sentiment"""
    model_path = os.path.join(MODEL_DIR, "sentiment_mllib_lr")

    env_vars = []
    for key in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                "POSTGRES_USER", "POSTGRES_PASSWORD", "MODEL_DIR",
                "SENTIMENT_TRAIN_TABLE"]:
        val = os.getenv(key, "")
        if val:
            env_vars.extend(["-e", f"{key}={val}"])

    cmd = [
        "docker", "exec",
        *env_vars,
        SPARK_CONTAINER,
        "/opt/bitnami/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "--jars", "/opt/spark/jars/postgresql-42.7.4.jar",
        "/opt/project/spark/ml_train_sentiment.py",
    ]

    context.log.info(f"Training model: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        context.log.warning(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise Exception(f"Training failed (exit {result.returncode})")

    context.log.info(f"✅ Model saved to {model_path}")
    return model_path


@op(ins={"model_path": In(str)}, tags={"module": "ml"})
def apply_model_to_toots(context: OpExecutionContext, model_path: str) -> None:
    """Applique le modèle aux toots"""
    env_vars = []
    for key in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                "POSTGRES_USER", "POSTGRES_PASSWORD"]:
        val = os.getenv(key, "")
        if val:
            env_vars.extend(["-e", f"{key}={val}"])

    env_vars.extend(["-e", f"MODEL_PATH={model_path}"])

    cmd = [
        "docker", "exec",
        *env_vars,
        SPARK_CONTAINER,
        "/opt/bitnami/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "--jars", "/opt/spark/jars/postgresql-42.7.4.jar",
        "/opt/project/spark/ml_apply_sentiment.py",
    ]

    context.log.info(f"Applying model: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        context.log.warning(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise Exception(f"Inference failed (exit {result.returncode})")

    context.log.info("✅ Model applied successfully")
