from dagster import sensor, RunRequest, SensorEvaluationContext
from dagster import DagsterRunStatus


@sensor(
    job_name="ml_pipeline",
    description="Démarre le ML après la fin du job d'ingestion"
)
def trigger_ml_after_ingestion(context: SensorEvaluationContext):
    # Récupère les derniers runs du job ingest_and_batch
    runs = context.instance.get_runs(
        filters={"job_name": "ingest_and_batch"},
        limit=1
    )

    if runs and runs[0].status == DagsterRunStatus.SUCCESS:
        last_run_id = runs[0].run_id

        # Vérifie si on a déjà lancé le ML pour ce run
        if context.cursor != last_run_id:
            context.update_cursor(last_run_id)
            yield RunRequest(run_key=last_run_id)