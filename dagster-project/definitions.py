import os
from dagster import Definitions, define_asset_job, AssetSelection
from assets import raw_toots, analytics_tables, sentiment_model, toots_with_sentiment

from jobs import ingest_then_batch, job_ml_sentiment
from schedules import daily_02h
from ressources import SparkSubmitResource, PgEnvResource
from sensors import trigger_ml_after_ingestion

all_assets = [raw_toots, analytics_tables, sentiment_model, toots_with_sentiment]

# JOBS
ingest_and_batch_job = define_asset_job(
    name="ingest_and_batch",
    selection=AssetSelection.assets(raw_toots, analytics_tables),
    description="Ingestion Mastodon → Aggregations batch"
)

ml_pipeline_job = define_asset_job(
    name="ml_pipeline",
    selection=AssetSelection.assets(sentiment_model, toots_with_sentiment),
    description="Entraînement modèle → Application sur toots"
)

full_pipeline_job = define_asset_job(
    name="full_pipeline",
    selection=AssetSelection.all(),
    description="Pipeline complet : Ingestion → Batch → ML → Inference"
)
####

defs = Definitions(
    assets=all_assets,
    jobs=[ingest_then_batch, job_ml_sentiment, ingest_and_batch_job, ml_pipeline_job, full_pipeline_job],
    sensors=[trigger_ml_after_ingestion],
    schedules=[daily_02h],
    resources={
        "spark": SparkSubmitResource(
            submit_bin=os.getenv("SPARK_SUBMIT_BIN", "/opt/bitnami/spark/bin/spark-submit"),
            master=os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077"),
            extra_args=os.getenv("SPARK_SUBMIT_EXTRA_ARGS", ""),
        ),
        "pg": PgEnvResource(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            db=os.getenv("POSTGRES_DB", "postgres"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        ),
    },
)