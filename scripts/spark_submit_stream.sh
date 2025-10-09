#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' config/.env | xargs)
docker compose -f compose/docker-compose.core.yml up -d streaming
