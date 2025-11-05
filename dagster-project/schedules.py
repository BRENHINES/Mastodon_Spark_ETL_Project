from dagster import ScheduleDefinition
from jobs import ingest_then_batch

daily_02h = ScheduleDefinition(
    job=ingest_then_batch,
    cron_schedule="0 2 * * *",  # tous les jours à 02:00
    execution_timezone="UTC",    # ou ton TZ
    name="schedule_batch_daily_02h",
)
