---
name: data-ingest
description: '获取A股行情与财务数据并转换为Qlib二进制格式。Use when: 下载股票数据、获取日线/财务数据、akshare、tushare、数据入库、qlib dump_bin、point-in-time数据、cn_data。'
argument-hint: '股票代码或数据区间'
---

# 数据获取与入库

## 目标
把 A 股数据（行情 + 财务）下载并整理成 Qlib 可用的二进制格式，供因子计算与回测使用。

## 数据源
- **akshare**：免费，无需 token，适合起步。
- **tushare**：需 token，财务数据质量更高，可作补充。
- **chenditc/investment_data**：现成的 Qlib 二进制数据（GitHub 每日 release），可作快速验证。

## 数据范围
- 区间：2016-01-01 ~ 2026-12-31。
- 字段：开高低收、成交量、成交额、vwap、换手率、市值、行业、财务指标。

## Procedure
1. 确认数据源（默认 akshare），检查依赖 `pip list`。
2. 下载日线数据，统一字段命名（`date/open/high/low/close/volume/amount/vwap/turnover`）。
3. 财务数据按 **point-in-time** 对齐（用公告日期，避免未来函数）。
4. 调用 Qlib `dump_bin` 转成二进制：`python scripts/dump_bin.py dump_all --csv_path <dir> --qlib_dir <dir> --include_fields open,high,low,close,volume,amount,vwap,turnover`。
5. 校验：随机抽样对比原始 CSV 与 Qlib 读取结果一致。

## 铁律
- 财务数据必须用 point-in-time（公告日），不能用报告期直接对齐。
- 转换后校验样本一致性，防止数据错位。

## 参考
- 项目约定见 [copilot-instructions.md](../../copilot-instructions.md)
- Qlib 官方：[chenditc/investment_data](https://github.com/chenditc/investment_data)
