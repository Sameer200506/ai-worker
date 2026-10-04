#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
./.venv/bin/pip install -r backend/requirements.txt
./.venv/bin/python -m uvicorn apps.sandbox.main:app --host 127.0.0.1 --port 3001 &
./.venv/bin/python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload &
cd apps/web
if [ ! -d node_modules ]; then npm install; fi
npm run dev
