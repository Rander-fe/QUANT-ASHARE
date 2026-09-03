"""Minimal deployment API for the existing research pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from main import STAGES

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "jobs"
API_TOKEN = os.getenv("QUANT_API_TOKEN", "").strip()
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="QUANT-ASHARE Research API", version="1.0.0")


class JobRequest(BaseModel):
    stage: str
    args: list[str] = Field(default_factory=list, max_length=32)


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QUANT_API_TOKEN is not configured",
        )
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_job(job_id: str, stage: str, args: list[str], log_path: Path) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(status="running", started_at=_utc_now())
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), stage, *args],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        final_status = "succeeded" if result.returncode == 0 else "failed"
        with JOBS_LOCK:
            JOBS[job_id].update(
                status=final_status,
                returncode=result.returncode,
                finished_at=_utc_now(),
            )
    except Exception as exc:  # pragma: no cover - last-resort job audit
        with JOBS_LOCK:
            JOBS[job_id].update(status="failed", error=str(exc), finished_at=_utc_now())


@app.get("/health")
def health() -> dict[str, Any]:
    checks = {
        "data": (ROOT / "data").is_dir(),
        "models": (ROOT / "models").is_dir(),
        "reports": (ROOT / "reports").is_dir(),
    }
    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}


@app.get("/stages", dependencies=[Depends(require_token)])
def stages() -> dict[str, dict[str, str]]:
    return {name: {"script": value[0], "description": value[1]} for name, value in STAGES.items()}


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_token)])
def create_job(request: JobRequest) -> dict[str, Any]:
    if request.stage not in STAGES:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {request.stage}")
    if any("\x00" in arg or len(arg) > 512 for arg in request.args):
        raise HTTPException(status_code=400, detail="Invalid stage argument")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    log_path = LOG_DIR / f"{job_id}.log"
    record = {
        "id": job_id,
        "stage": request.stage,
        "args": request.args,
        "status": "queued",
        "created_at": _utc_now(),
        "log_path": str(log_path.relative_to(ROOT)),
    }
    with JOBS_LOCK:
        JOBS[job_id] = record
    threading.Thread(
        target=_run_job,
        args=(job_id, request.stage, request.args, log_path),
        daemon=True,
    ).start()
    return record


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_token)])
def get_job_log(job_id: str, tail: int = 200) -> dict[str, Any]:
    tail = min(max(tail, 1), 2000)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        log_path = ROOT / job["log_path"]
    if not log_path.exists():
        return {"id": job_id, "lines": []}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"id": job_id, "lines": lines[-tail:]}
