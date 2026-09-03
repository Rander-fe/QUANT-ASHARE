#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANT_ASHARE_PROJECT_DIR:-/opt/quant-ashare}"
cd "$PROJECT_DIR"

git pull --ff-only
docker compose build research-api
docker compose up -d research-api
docker compose ps

