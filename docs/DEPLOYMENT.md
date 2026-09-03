# QUANT-ASHARE 部署指南

本部署层借鉴 QuantMind 的 Docker Compose、环境变量、持久卷、健康检查和幂等更新方式，保留本项目现有研究流水线，不引入 QuantMind 的前端、数据库及交易服务。

## 推荐环境

- Ubuntu 22.04/24.04
- Docker Engine + Docker Compose plugin + `rsync`
- 8 核 / 16 GB 起；训练深度模型建议 64 GB 内存和 NVIDIA GPU
- 项目当前数据、模型和报告约 50 GB，建议至少预留 150 GB

## 部署

```bash
git clone https://github.com/Rander-fe/QUANT-ASHARE.git /opt/quant-ashare-src
cd /opt/quant-ashare-src
sudo QUANT_ASHARE_PROJECT_DIR=/opt/quant-ashare bash deploy/deploy.sh
```

脚本首次运行会创建 `/opt/quant-ashare/.env`，并生成 API Token。把现有数据分别同步到：

```text
/opt/quant-ashare/data
/opt/quant-ashare/models
/opt/quant-ashare/reports
/opt/quant-ashare/mlruns
```

在 `.env` 中把 `QLIB_DATA_HOST_PATH` 设置为宿主机 Qlib `cn_data` 的绝对路径，然后重启：

```bash
cd /opt/quant-ashare
docker compose up -d research-api
```

## 使用

健康检查和 API 文档：

```bash
curl http://127.0.0.1:8000/health
# 浏览器访问 http://SERVER_IP:8000/docs
```

运行一个已有流水线阶段：

```bash
TOKEN=$(sed -n 's/^QUANT_API_TOKEN=//p' .env)
curl -X POST http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer ${TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"stage":"build_factors","args":[]}'
```

也可以直接使用 CLI 容器：

```bash
docker compose run --rm research-cli list
docker compose run --rm research-cli build_factors
docker compose run --rm research-cli train_lgb
```

## 更新与回滚

```bash
sudo QUANT_ASHARE_PROJECT_DIR=/opt/quant-ashare bash deploy/update.sh
```

更新脚本不会删除挂载的 `data/`、`models/`、`reports/`、`mlruns/` 和 `logs/`。本次本地改造前的完整备份位于 `C:\QUANT-ASHARE_BACKUP_20260831_173000`。

停止服务但保留数据：

```bash
docker compose down
```

不要使用 `docker compose down -v`，除非明确要删除卷数据。
