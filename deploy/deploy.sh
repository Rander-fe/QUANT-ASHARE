#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${QUANT_ASHARE_PROJECT_DIR:-/opt/quant-ashare}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Engine and the Compose plugin first." >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required (Ubuntu: sudo apt-get install -y rsync)." >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR"
if [[ "$SOURCE_DIR" != "$PROJECT_DIR" ]]; then
  rsync -a --delete \
    --exclude '.env' --exclude 'data/' --exclude 'models/' \
    --exclude 'reports/' --exclude 'mlruns/' --exclude 'logs/' --exclude 'tmp/' \
    "$SOURCE_DIR/" "$PROJECT_DIR/"
fi

cd "$PROJECT_DIR"
mkdir -p data/raw data/processed data/qlib models reports mlruns logs
if [[ ! -f .env ]]; then
  cp .env.deploy.example .env
fi
current_token="$(sed -n 's/^QUANT_API_TOKEN=//p' .env | tail -n 1)"
if [[ -z "$current_token" || "$current_token" == "CHANGE_ME" ]]; then
  token="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  if grep -q '^QUANT_API_TOKEN=' .env; then
    sed -i "s/^QUANT_API_TOKEN=.*/QUANT_API_TOKEN=${token}/" .env
  else
    printf '\nQUANT_API_TOKEN=%s\n' "$token" >> .env
  fi
  echo "Configured a generated API token in .env."
fi
chmod 600 .env

set -a
# shellcheck disable=SC1091
source .env
set +a

docker compose build research-api
docker compose up -d research-api

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:"${API_PORT:-8000}"/health >/dev/null; then
    docker compose ps
    echo "QUANT-ASHARE is healthy."
    exit 0
  fi
  sleep 2
done

docker compose logs --tail 100 research-api
echo "Deployment did not become healthy in time." >&2
exit 1
