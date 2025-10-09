# Mastodon Streaming with Spark (Kafka → Spark → Postgres → MLlib)

## Stack
- Kafka + ZooKeeper
- Spark (Structured Streaming + Batch)
- PostgreSQL
- Dagster (orchestration)
- Prometheus + Grafana (observabilité)
- Jupyter (viz)

## Run (high-level)
1) cp config/.env.example config/.env
2) docker compose -f compose/docker-compose.core.yml up -d --build
3) bash scripts/create_topics.sh
4) bash scripts/spark_submit_stream.sh
5) docker compose -f compose/docker-compose.addons.yml up -d --build
