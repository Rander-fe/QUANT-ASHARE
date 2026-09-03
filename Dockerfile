FROM python:3.9-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ml.txt requirements-deploy.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements-ml.txt -r requirements-deploy.txt

COPY . /app
RUN chmod +x /app/deploy/entrypoint.sh /app/deploy/healthcheck.sh \
    && mkdir -p /app/data/raw /app/data/processed /app/models /app/reports /app/logs

EXPOSE 8000
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["api"]

