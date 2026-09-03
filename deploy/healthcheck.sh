#!/usr/bin/env sh
set -eu
curl --fail --silent --max-time 5 http://127.0.0.1:8000/health >/dev/null

