# dagster-project/assets.py
import os
import subprocess
from dagster import (
    asset, AssetExecutionContext, MetadataValue,
    AssetIn, AssetOut,
)

PROJECT_ROOT = "/opt/project"
SPARK_CONTAINER = os.getenv("SPARK_CONTAINER", "spark-mastodon-core-spark-master-1")

# Helpers communs
def _copy_env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    if extra:
        for k, v in extra.items():
            env[str(k)] = str(v)
    return env

# -----------------------------
# ASSET 1: Stream -> table "toots"
# -----------------------------
@asset(
    name="raw_toots",
    description="Ingestion Mastodon -> Postgres (table `toots`). Fenêtré (durée / max toots).",
    io_manager_key="io_manager",  # par défaut; pas utilisé ici mais OK
    # pas d'inputs: c'est une source/ingestion
)
def raw_toots(context: AssetExecutionContext):
    context.log.info("🚀 Starting Mastodon stream ingestion...")

    duration = int(os.getenv("STREAM_DURATION_SECS", "350"))
    max_toots = int(os.getenv("STREAM_MAX_TOOTS", "1500"))

    context.log.info(f"📊 Config: duration={duration}s, max_toots={max_toots}")

    cmd = [
        "python", "/opt/project/spark/streaming_to_postgres.py",
        "--duration-sec", str(duration),
        "--max-toots", str(max_toots),
    ]

    env = _copy_env()
    context.log.info(f"[STREAM] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)

    context.log.info("✅ Stream completed!")
    context.log.info(f"STDOUT:\n{result.stdout}")

    # Log console
    if result.stdout:
        context.log.info(result.stdout)
    if result.stderr:
        # network timeouts, etc. apparaîtront ici
        context.log.warning(result.stderr)

    if result.returncode != 0:
        raise Exception(f"[STREAM] exit={result.returncode}")

    # métadonnées visibles dans l'UI
    context.add_output_metadata({
        "duration_secs": duration,
        "max_toots": max_toots,
        "stdout": MetadataValue.text(result.stdout[-8000:] if result.stdout else ""),
    })


# ---------------------------------------
# ASSET 2: Batch aggregations -> 3 tables
# ---------------------------------------
@asset(
    name="analytics_tables",
    description="Batch Spark: calcule daily_counts, top_hashtags, user_activity dans Postgres.",
    ins={"raw_toots": AssetIn("raw_toots")},  # ✅ Nom doit correspondre au paramètre
)
def analytics_tables(context: AssetExecutionContext, raw_toots):  # ✅ Ajout du paramètre
    """
    Note: raw_toots n'est pas vraiment utilisé dans la fonction,
    mais il crée la dépendance pour que Dagster exécute dans le bon ordre
    """
    env_vars = []
    for key in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_TABLE",
                "SPARK_MASTER"]:
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
        "/opt/project/spark/batch_aggregations.py",
    ]

    context.log.info(f"[BATCH] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.stdout:
        context.log.info(result.stdout)
    if result.stderr:
        context.log.warning(result.stderr)
    if result.returncode != 0:
        raise Exception(f"[BATCH] exit={result.returncode}")

    context.add_output_metadata({
        "tables": MetadataValue.json(["daily_counts", "top_hashtags", "user_activity"]),
        "stdout": MetadataValue.text(result.stdout[-8000:] if result.stdout else ""),
    })


# ----------------------------------------
# ASSET 3: Model training -> modèle sur /models
# ----------------------------------------
@asset(
    name="sentiment_model",
    description="Entraîne un modèle LR (Sentiment140) et l'enregistre dans MODEL_DIR/MODEL_NAME.",
    ins={"analytics_tables": AssetIn("analytics_tables")},
)
def sentiment_model(context: AssetExecutionContext, analytics_tables):
    """
    analytics_tables n'est pas utilisé, mais crée la dépendance
    """
    model_dir = os.getenv("MODEL_DIR", "/models")
    model_name = os.getenv("MODEL_NAME", "sentiment_lr_en")
    model_path = os.path.join(model_dir, model_name)

    # ✅ INSTALLER LES DÉPENDANCES DANS LE CONTENEUR SPARK
    context.log.info("[INSTALL] Installation des dépendances dans Spark...")
    install_cmd = [
        "docker", "exec",
        SPARK_CONTAINER,
        "pip", "install",
        "numpy>=1.21,<2.0",
        "scipy>=1.7,<1.12",
        "pandas>=1.3,<2.0",
        "scikit-learn>=1.0,<1.3"
    ]

    install_result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=180)

    if install_result.returncode == 0:
        context.log.info("✅ Dépendances installées dans Spark")
    else:
        context.log.error(f"❌ Erreur installation: {install_result.stderr}")
        raise Exception(f"Échec installation: {install_result.stderr}")

    # Construction des variables d'environnement
    env_vars = []
    for key in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                "POSTGRES_USER", "POSTGRES_PASSWORD", "MODEL_DIR",
                "SENTIMENT140_CSV", "MODEL_NAME"]:
        val = os.getenv(key, "")
        if val:
            env_vars.extend(["-e", f"{key}={val}"])
            context.log.info(f"✅ {key}={val}")
        else:
            context.log.warning(f"⚠️ {key} is not set!")

    # ✅ CORRECTION : env_vars AVANT le nom du conteneur
    cmd = [
        "docker", "exec",
        *env_vars,  # ✅ Les options -e viennent en premier
        SPARK_CONTAINER,  # ✅ Puis le nom du conteneur
        "/opt/bitnami/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "--jars", "/opt/spark/jars/postgresql-42.7.4.jar",
        "/opt/project/spark/train_sentiment_mllib.py",
    ]

    context.log.info(f"[TRAIN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        context.log.info(result.stdout)
    if result.stderr:
        context.log.warning(result.stderr)
    if result.returncode != 0:
        raise Exception(f"[TRAIN] exit={result.returncode}\nSTDERR:\n{result.stderr}")

    context.add_output_metadata({
        "model_dir": model_dir,
        "model_name": model_name,
        "model_path": model_path,
        "stdout": MetadataValue.text(result.stdout[-8000:] if result.stdout else ""),
    })

    return {"model_path": model_path}


# ----------------------------------------------------
# ASSET 4: Apply model -> table `toots_with_sentiment`
# ----------------------------------------------------
@asset(
    name="toots_with_sentiment",
    description="Applique le modèle sur les toots et écrit dans Postgres (table `toots_with_sentiment`).",
    ins={"model": AssetIn("sentiment_model")},
)
def toots_with_sentiment(context: AssetExecutionContext, model):
    """
    `model` est le retour de l’asset précédent (dict), on récupère le chemin depuis metadata/retour.
    """
    # récupération du chemin
    model_path = None
    # 1) via le retour de l’asset précédent
    if isinstance(model, dict) and "model_path" in model:
        model_path = model["model_path"]
    # 2) fallback via env si jamais
    if not model_path:
        model_dir = os.getenv("MODEL_DIR", "/models")
        model_name = os.getenv("MODEL_NAME", "sentiment_lr_en")
        model_path = os.path.join(model_dir, model_name)

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
        "/opt/project/spark/apply_sentiment_batch.py",
    ]

    context.log.info(f"[APPLY] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        context.log.info(result.stdout)
    if result.stderr:
        context.log.warning(result.stderr)
    if result.returncode != 0:
        raise Exception(f"[APPLY] exit={result.returncode}")

    context.add_output_metadata({
        "model_path": model_path,
        "output_table": "toots_with_sentiment",
        "stdout": MetadataValue.text(result.stdout[-8000:] if result.stdout else ""),
    })
