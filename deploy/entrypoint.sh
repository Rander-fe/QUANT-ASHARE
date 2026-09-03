#!/usr/bin/env sh
set -eu

case "${1:-api}" in
  api)
    exec uvicorn deploy.service:app \
      --host "${API_HOST:-0.0.0.0}" \
      --port "${API_PORT:-8000}" \
      --workers "${API_WORKERS:-1}"
    ;;
  stage)
    shift
    exec python /app/main.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

