# 文本数据采集 MVP

该模块先保存原始新闻、政策和公告信息，不执行情绪打分，也不直接进入模型。

## 使用

查看所有数据源：

```powershell
C:/Users/haoran/miniconda3/envs/rqalpha/python.exe main.py collect_text --list-sources
```

采集当前启用的数据源：

```powershell
C:/Users/haoran/miniconda3/envs/rqalpha/python.exe main.py collect_text
```

原始文档保存在：

```text
data/raw/text/{news|policy|announcement}/YYYY/MM/DD/{document_id}.json
```

采集状态、去重索引和失败记录保存在：

```text
data/processed/text/collector_state.sqlite3
```

每条原始记录保留 `published_at`、`first_seen_at`、`collected_at`、原文URL、内容哈希和来源字段。重复运行不会重复保存同一来源的相同内容。

## 当前数据源状态

| 数据源 | 默认状态 | 说明 |
|---|---:|---|
| 中国政府网最新政策JSON | 启用 | 公开接口，无账号要求 |
| 人社部政策法规/解读RSS | 关闭 | 当前返回JavaScript反爬页面而非XML |
| Tushare `anns_d`公告 | 关闭 | 当前Token没有接口权限 |
| Tushare `npr`国家政策库 | 关闭 | 当前Token没有接口权限 |
| 财联社/同花顺RSSHub | 关闭 | 需要先部署RSSHub |

修改 [text_sources.json](../config/text_sources.json) 中的 `enabled` 可以启停数据源。不要在未验证权限或服务地址前批量开启。

## 当前边界

- 中国政府网源目前保存标题、发布日期与原文链接，不抓取全文。
- Tushare公告适配器保留PDF链接和股票代码，不自动下载PDF。
- RSS适配器保存feed提供的摘要，不绕过网站反爬或付费限制。
- 文档尚未进行股票映射、行业映射、事件分类或情绪分析。
- 正式回测必须以 `first_seen_at` 计算信息可用日，并至少延迟到下一交易日使用。

下一阶段应实现文本富化表：实体映射、事件标签、政策行业映射和情绪评分；原始JSON文件保持不变。

