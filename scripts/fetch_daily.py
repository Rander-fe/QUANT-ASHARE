# -*- coding: utf-8 -*-
"""从 Qlib 二进制行情库读取 A 股日线数据，导出为长表 parquet。

数据源：`~/.qlib/qlib_data/cn_data`（chenditc/investment_data release，6135 只，
2000-01-04 ~ 2026-08-13）。本脚本按交易所前缀（sh/sz/bj）分批读取，落盘为长表：

    data/raw/daily/daily_sh.parquet / daily_sz.parquet / daily_bj.parquet

长表列：symbol, date, open, high, low, close, volume, amount, vwap, factor, adjclose

说明：
- `$close` 为前复权收盘价（qlib 存储口径），`$factor` 为复权因子，
  `$adjclose` 为后复权收盘价；复权口径细节见 knowledge/ 与预处理环节。
- 区间默认取 settings.DATA_START ~ DATA_END，可 --start/--end 覆盖。
- 股票清单默认 instruments/all.txt，可 --instruments 覆盖（格式：symbol\\tlist_date\\tdelist_date）。

用法：
    python scripts/fetch_daily.py                       # 全市场，全区间
    python scripts/fetch_daily.py --limit 100           # 只读前 100 只（快速验证）
    python scripts/fetch_daily.py --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 静音 gym 0.26.2 每次 import 时向 stderr 打印的 "Gym has been unmaintained" 刷屏
# （qlib 的策略/数据链路会反复 import gym，不静音会污染日志与 exit code）。
try:
    import gym_notices.notices as _gn
    _gn.notices = {}
except Exception:  # noqa: BLE001
    pass

# 允许以 `python scripts/...` 运行时导入项目根目录下的 config 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    DAILY_RAW_DIR,
    DATA_END,
    DATA_START,
    QLIB_PROVIDER_URI,
)

try:
    import qlib  # noqa: E402
    from qlib.data import D  # noqa: E402
except ModuleNotFoundError as _exc:  # pragma: no cover
    raise SystemExit(
        "[error] 当前 Python 环境未安装 qlib。\n"
        "请用 rqalpha 环境运行（qlib 仅支持 py3.8~3.11，不要用 py3.13）：\n"
        "    C:/Users/haoran/miniconda3/envs/rqalpha/python.exe scripts/fetch_daily.py\n"
        f"原始错误：{_exc}"
    ) from _exc

# 导出的字段（qlib 表达式名 -> 落盘列名）
DAILY_FIELDS = [
    ("$open", "open"),
    ("$high", "high"),
    ("$low", "low"),
    ("$close", "close"),
    ("$volume", "volume"),
    ("$amount", "amount"),
    ("$vwap", "vwap"),
    ("$factor", "factor"),
    ("$adjclose", "adjclose"),
]

# 交易所前缀 -> 落盘文件名（去除 bj 北交所外的指数，仅保留个股按前缀归档）
EXCHANGE_PREFIXES = ("sh", "sz", "bj")


def init_qlib(provider_uri: str = QLIB_PROVIDER_URI) -> None:
    """初始化 qlib 数据环境（集中配置，供 run() 调用）。

    配置项：
    - provider_uri：二进制行情库路径，~ 统一展开为绝对路径（Windows 兼容）。
    - region：中国市场 'cn'。
    """
    uri = str(Path(provider_uri).expanduser())
    print(f"[info] 初始化 qlib，数据源：{uri}")
    qlib.init(provider_uri=uri, region="cn")


def load_instruments(provider_uri: str, instruments_arg: str) -> list[str]:
    """读取股票清单，返回有序 symbol 列表。

    instruments_arg 为 "all" 时读 instruments/all.txt；否则视为文件路径。
    all.txt 每行：symbol<TAB>list_date<TAB>delist_date，只取 symbol 列。
    """
    root = Path(provider_uri).expanduser()
    if instruments_arg == "all":
        path = root / "instruments" / "all.txt"
    else:
        path = Path(instruments_arg).expanduser()

    symbols = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            symbol = line.split("\t")[0].strip()
            if symbol:
                symbols.append(symbol)
    return symbols


def _group_by_exchange(symbols: list[str]) -> dict[str, list[str]]:
    """按交易所前缀（sh/sz/bj）分组，返回 {prefix: [symbols]}。"""
    groups: dict[str, list[str]] = {p: [] for p in EXCHANGE_PREFIXES}
    for s in symbols:
        prefix = s[:2].lower()
        if prefix in groups:
            groups[prefix].append(s)
    return groups


def fetch_batch(instruments: list[str], fields: list[str], start: str, end: str) -> pd.DataFrame:
    """用 D.features 读取一批股票，返回长表 DataFrame（symbol, date, ...）。"""
    df = D.features(instruments, fields, start_time=start, end_time=end, freq="day")
    if df.empty:
        return df
    df = df.reset_index()
    # index 名称为 instrument / datetime，统一重命名
    rename = {c: ("date" if c == "datetime" else "symbol" if c == "instrument" else c) for c in df.columns}
    df = df.rename(columns=rename)
    if "date" in df.columns and "symbol" in df.columns:
        return df[["symbol", "date"] + [c for c in df.columns if c not in ("symbol", "date")]]
    return df


def run(
    instruments_arg: str,
    start: str,
    end: str,
    limit: int | None,
    batch_size: int,
    out_dir: Path,
) -> None:
    """主流程：读清单 -> 按交易所分组 -> 分批读取 -> 落盘 parquet。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_instruments(QLIB_PROVIDER_URI, instruments_arg)
    if limit is not None:
        symbols = symbols[:limit]
    if not symbols:
        print("[warn] 股票清单为空，退出。")
        return

    init_qlib()

    expr_fields = [f for f, _ in DAILY_FIELDS]
    groups = _group_by_exchange(symbols)

    total_rows = 0
    for prefix in EXCHANGE_PREFIXES:
        syms = groups[prefix]
        if not syms:
            continue
        out_path = out_dir / f"daily_{prefix}.parquet"
        frames: list[pd.DataFrame] = []
        n_batches = (len(syms) + batch_size - 1) // batch_size
        print(f"[info] {prefix} 共 {len(syms)} 只，分 {n_batches} 批读取 ...")
        for i in range(n_batches):
            batch = syms[i * batch_size:(i + 1) * batch_size]
            df = fetch_batch(batch, expr_fields, start, end)
            if not df.empty:
                frames.append(df)
            print(f"    [{prefix}] batch {i + 1}/{n_batches}：{batch[0]}~{batch[-1]} -> {len(df)} 行")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            # 统一列名去掉 $ 前缀（D.features 返回的列名即为去掉 $ 后的字段名）
            combined.columns = [c.lstrip("$") for c in combined.columns]
            combined.to_parquet(out_path, index=False)
            total_rows += len(combined)
            print(f"[done] {out_path}：{len(combined)} 行，{combined['symbol'].nunique()} 只")
        else:
            print(f"[warn] {prefix} 无数据，跳过。")

    print(f"[done] 全部落盘完成，共 {total_rows} 行 -> {out_dir}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="从 Qlib 读取 A 股日线并导出 parquet")
    parser.add_argument("--instruments", default="all", help="股票清单：all 或文件路径（默认 all）")
    parser.add_argument("--start", default=DATA_START, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=DATA_END, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="只读前 N 只（快速验证用）")
    parser.add_argument("--batch-size", type=int, default=500, help="每批读取的股票数（默认 500）")
    parser.add_argument("--out-dir", default=str(DAILY_RAW_DIR), help="落盘目录")
    args = parser.parse_args()

    run(
        instruments_arg=args.instruments,
        start=args.start,
        end=args.end,
        limit=args.limit,
        batch_size=args.batch_size,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
