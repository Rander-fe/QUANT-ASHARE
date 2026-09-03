"""Fetch and validate historical SW2021 industry membership from Tushare REST API."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_tushare_token

API = "https://api.tushare.pro"
RAW_DIR = ROOT / "data" / "raw" / "industry"
OUT_DIR = ROOT / "data" / "processed" / "industry"
SW2021_EFFECTIVE = pd.Timestamp("2021-12-13")


def api_call(token: str, api_name: str, params: dict, fields: str = "") -> pd.DataFrame:
    payload = json.dumps({"api_name": api_name, "token": token, "params": params, "fields": fields}).encode()
    request = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Tushare HTTP {exc.code}") from None
    if result.get("code") != 0:
        raise RuntimeError(f"Tushare {api_name} failed: {result.get('msg', 'unknown error')}")
    data = result.get("data") or {}
    return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])


def to_symbol(ts_code: str) -> str:
    code, exchange = str(ts_code).split(".")
    return f"{exchange}{code}"


def main() -> int:
    token = get_tushare_token()
    if not token:
        raise RuntimeError("未配置TUSHARE_TOKEN")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classify = api_call(token, "index_classify", {"level": "L1", "src": "SW2021"})
    if classify.empty:
        raise RuntimeError("index_classify返回空表")
    classify.to_parquet(RAW_DIR / "sw2021_classify.parquet", index=False)
    code_col = "index_code" if "index_code" in classify else "industry_code"

    new_member_path = RAW_DIR / "sw2021_members_all.parquet"
    if new_member_path.exists():
        members = pd.read_parquet(new_member_path)
    else:
        parts = []
        for code in classify[code_col].dropna().astype(str).drop_duplicates():
            for is_new in ("Y", "N"):
                part = api_call(token, "index_member_all", {"l1_code": code, "is_new": is_new})
                if len(part) >= 2000:
                    raise RuntimeError(f"{code}/{is_new}返回2000行，可能被接口上限截断")
                parts.append(part)
            print(f"fetched SW2021 {code}", flush=True)
        members = pd.concat(parts, ignore_index=True).drop_duplicates()
    required = {"l1_code", "l1_name", "ts_code", "in_date", "out_date", "is_new"}
    if not required <= set(members):
        raise RuntimeError(f"行业成员缺少字段: {sorted(required-set(members))}")
    members.to_parquet(new_member_path, index=False)

    classify_old = api_call(token, "index_classify", {"level": "L1", "src": "SW2014"})
    classify_old.to_parquet(RAW_DIR / "sw2014_classify.parquet", index=False)
    old_names = classify_old.set_index("index_code")["industry_name"].to_dict()
    old_member_path = RAW_DIR / "sw2014_members_all.parquet"
    if old_member_path.exists():
        old_members = pd.read_parquet(old_member_path)
    else:
        old_parts = []
        for code in classify_old["index_code"].dropna().astype(str).drop_duplicates():
            part = api_call(token, "index_member", {"index_code": code})
            if len(part) >= 2000:
                raise RuntimeError(f"SW2014 {code}返回2000行，可能被截断")
            part["l1_code"], part["l1_name"] = code, old_names.get(code)
            part = part.rename(columns={"con_code": "ts_code"})
            old_parts.append(part)
            print(f"fetched SW2014 {code}", flush=True)
        old_members = pd.concat(old_parts, ignore_index=True).drop_duplicates()
        old_members.to_parquet(old_member_path, index=False)

    intervals = members.copy()
    intervals["classification_version"] = "SW2021"
    old_members["classification_version"] = "SW2014"
    intervals = pd.concat([old_members, intervals], ignore_index=True, sort=False)
    intervals["symbol"] = intervals["ts_code"].map(to_symbol)
    intervals["start_date"] = pd.to_datetime(intervals["in_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["out_date"], errors="coerce")
    intervals = intervals.dropna(subset=["symbol", "l1_code", "start_date"])
    bad = intervals["end_date"].notna() & (intervals["end_date"] <= intervals["start_date"])
    invalid_source_intervals = int(bad.sum())
    intervals = intervals.loc[~bad].copy()
    old = intervals["classification_version"].eq("SW2014")
    intervals.loc[old, "end_date"] = intervals.loc[old, "end_date"].fillna(SW2021_EFFECTIVE).clip(upper=SW2021_EFFECTIVE)
    intervals.loc[~old, "start_date"] = intervals.loc[~old, "start_date"].clip(lower=SW2021_EFFECTIVE)
    intervals = intervals.loc[intervals["end_date"].isna() | intervals["start_date"].lt(intervals["end_date"])]
    intervals = intervals.rename(columns={"l1_code": "industry_code", "l1_name": "industry_name"})
    keep = ["symbol", "industry_code", "industry_name", "start_date", "end_date", "is_new", "ts_code", "classification_version"]
    intervals = intervals[keep].drop_duplicates().sort_values(["symbol", "start_date", "industry_code"])
    overlap = 0
    for _, group in intervals.groupby("symbol", sort=False):
        previous_end = group["end_date"].shift().fillna(pd.Timestamp.max)
        overlap += int((group["start_date"] < previous_end).iloc[1:].sum())
    if overlap:
        raise RuntimeError(f"发现{overlap}个股票行业区间重叠，禁止自动写入PIT表")
    intervals.to_parquet(OUT_DIR / "sw_l1_membership_intervals.parquet", index=False)
    report = {
        "status": "ready", "source": "Tushare index_classify + index_member_all",
        "classification": "SW2014 L1 before 2021-12-13; SW2021 L1 on/after 2021-12-13", "rows": len(intervals),
        "symbols": intervals["symbol"].nunique(), "industries": intervals["industry_code"].nunique(),
        "start": str(intervals["start_date"].min().date()),
        "open_intervals": int(intervals["end_date"].isna().sum()), "overlaps": overlap,
        "invalid_source_intervals_excluded": invalid_source_intervals,
        "interval_semantics": "start_date <= date < end_date; null end_date means active",
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
