# dagster-project/assets.py
import os
import subprocess
from dagster import (
    asset, AssetExecutionContext, MetadataValue,
    AssetIn, AssetOut, MaterializeResult
)
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = "/opt/project"
SPARK_CONTAINER = os.getenv("SPARK_CONTAINER", "spark-mastodon-core-spark-master-1")

REPORT_DIR = os.getenv("REPORT_DIR", "/reports")
TABLE      = os.getenv("SENTIMENT_TABLE", "toots_with_sentiment")

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
    
def _conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

@asset(
    name="sentiment_visualizations",
    description="Génère des visualisations à partir de toots_with_sentiment et sauvegarde des PNG.",
    ins={"visuals": AssetIn("toots_with_sentiment")},
)
def sentiment_visualizations(context: AssetExecutionContext, visuals) -> MaterializeResult:
    _ensure_dir(REPORT_DIR)

    with _conn() as conn:
        # 1) Série quotidienne
        q_daily = f"""
        SELECT date_trunc('day', ts) AS day,
               COUNT(*)::int AS n,
               AVG(CASE WHEN sentiment_label='positive' THEN 1 ELSE 0 END)::float AS pos_rate,
               AVG(sentiment_score)::float AS avg_score
        FROM {TABLE}
        WHERE ts IS NOT NULL
        GROUP BY 1
        ORDER BY 1;
        """
        df_daily = pd.read_sql(q_daily, conn)

        # 2) Histogramme des scores
        q_scores = f"SELECT sentiment_score FROM {TABLE};"
        df_scores = pd.read_sql(q_scores, conn)

        # 3) Langues (top 10)
        q_lang = f"""
        SELECT COALESCE(NULLIF(lang,''),'und') AS lang, COUNT(*)::int AS n
        FROM {TABLE}
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 10;
        """
        df_lang = pd.read_sql(q_lang, conn)

        # 4) Top users (seuil de support)
        q_users = f"""
        SELECT username,
               COUNT(*)::int AS n,
               AVG(CASE WHEN sentiment_label='positive' THEN 1 ELSE 0 END)::float AS pos_rate
        FROM {TABLE}
        WHERE username IS NOT NULL AND username <> ''
        GROUP BY username
        HAVING COUNT(*) >= 20
        ORDER BY pos_rate DESC, n DESC
        LIMIT 20;
        """
        df_users = pd.read_sql(q_users, conn)

    saved = {}

    # --- Plot 1: Série quotidienne ---
    if not df_daily.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(df_daily["day"], df_daily["n"])
        ax1.set_title("Toots / jour")
        ax1.set_xlabel("Jour")
        ax1.set_ylabel("Volume")

        # deuxième axe pour le taux de positifs (0..1)
        ax2 = ax1.twinx()
        ax2.plot(df_daily["day"], df_daily["pos_rate"], marker="o")
        ax2.set_ylabel("Taux positifs")

        p = os.path.join(REPORT_DIR, "sentiment_daily.png")
        _savefig(fig, p)
        saved["sentiment_daily"] = MetadataValue.path(p)

    # --- Plot 2: Histogramme des scores ---
    if not df_scores.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_scores["sentiment_score"].clip(0,1), bins=30)
        ax.set_title("Distribution des scores de sentiment")
        ax.set_xlabel("score positif (0..1)")
        ax.set_ylabel("fréquence")

        p = os.path.join(REPORT_DIR, "sentiment_score_hist.png")
        _savefig(fig, p)
        saved["sentiment_score_hist"] = MetadataValue.path(p)

    # --- Plot 3: Langues (top 10) ---
    if not df_lang.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(df_lang["lang"], df_lang["n"])
        ax.set_title("Top 10 langues")
        ax.set_xlabel("lang")
        ax.set_ylabel("toots")
        ax.tick_params(axis='x', rotation=45)

        p = os.path.join(REPORT_DIR, "lang_top10.png")
        _savefig(fig, p)
        saved["lang_top10"] = MetadataValue.path(p)

    # --- Plot 4: Top users (≥20 toots) par taux positif ---
    if not df_users.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        dfu = df_users.sort_values(["pos_rate", "n"], ascending=[True, True])
        ax.barh(dfu["username"], dfu["pos_rate"])
        ax.set_title("Top users par taux de positifs (≥20 toots)")
        ax.set_xlabel("taux positifs")
        ax.set_ylabel("username")

        p = os.path.join(REPORT_DIR, "top_users.png")
        _savefig(fig, p)
        saved["top_users"] = MetadataValue.path(p)

    # Résumé dans les métadonnées Dagster (tu peux cliquer les paths dans Dagit)
    return MaterializeResult(
        metadata={
            "output_dir": MetadataValue.path(REPORT_DIR),
            **saved
        }
    )
