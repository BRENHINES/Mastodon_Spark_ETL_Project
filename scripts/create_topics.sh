#!/usr/bin/env bash
set -euo pipefail
TOPIC=${1:-mastodon_stream}
PARTITIONS=${2:-3}
RETENTION=${3:-86400000}
docker compose -f compose/docker-compose.core.yml exec -T kafka \
  kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists \
  --topic "$TOPIC" --partitions "$PARTITIONS" --replication-factor 1 \
  --config retention.ms="$RETENTION"
