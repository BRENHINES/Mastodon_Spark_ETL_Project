from dagster import job
from ops import run_stream_for_window, run_batch_aggregations, train_sentiment_model, apply_model_to_toots

@job
def ingest_then_batch():
    run_batch_aggregations(start=run_stream_for_window())

@job(name="job_ml_sentiment")
def job_ml_sentiment():
    model_path = train_sentiment_model()
    apply_model_to_toots(model_path)
