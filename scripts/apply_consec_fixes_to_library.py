"""备份正式因子库，只替换 CONSEC_UP3/5；失败时恢复原文件。"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, REPORTS_DIR

BATCH_SIZE = 100_000
REPLACEMENTS = {"CONSEC_UP3": "CONSEC_UP3_V2", "CONSEC_UP5": "CONSEC_UP5_V2"}


def _next_batch(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def main() -> int:
    official = (DATA_PROCESSED / "factors.parquet").resolve()
    v2_path = (DATA_PROCESSED / "reversal_momentum_fixes_v2.parquet").resolve()
    backup_dir = (DATA_PROCESSED / "backups" / "factor_library").resolve()
    workspace = DATA_PROCESSED.resolve()
    if workspace not in official.parents or workspace not in v2_path.parents or workspace not in backup_dir.parents:
        raise RuntimeError("路径越出项目 data/processed，拒绝执行")
    if not official.exists() or not v2_path.exists():
        raise FileNotFoundError("缺少正式因子库或V2验证文件")

    old_meta = pq.ParquetFile(official).metadata
    v2_meta = pq.ParquetFile(v2_path).metadata
    if old_meta.num_rows != v2_meta.num_rows:
        raise RuntimeError("正式库与V2行数不同，拒绝替换")
    required_free = int(official.stat().st_size * 0.90)
    if shutil.disk_usage(official.parent).free < required_free:
        raise RuntimeError("磁盘余量不足以安全重建正式库")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"factors_before_consec_v2_{timestamp}.parquet"
    building = official.with_suffix(".parquet.building")
    if backup.exists() or building.exists():
        raise FileExistsError("备份名或临时文件已存在，拒绝覆盖")

    # 同盘原子重命名：原文件立即成为完整、可恢复的备份，不复制13GB。
    os.replace(official, backup)
    writer = None
    old_pf = None
    rows_written = 0
    v2_frame = pd.read_parquet(v2_path, columns=["symbol", "date"] + list(REPLACEMENTS.values()))
    v2_frame["date"] = pd.to_datetime(v2_frame["date"])
    v2_frame = v2_frame.set_index(["symbol", "date"], verify_integrity=True)
    v2_index = v2_frame.index
    used_v2_rows = np.zeros(len(v2_frame), dtype=bool)
    try:
        old_pf = pq.ParquetFile(backup)
        old_iter = old_pf.iter_batches(batch_size=BATCH_SIZE)
        while True:
            old_batch = _next_batch(old_iter)
            if old_batch is None:
                break
            old_table = pa.Table.from_batches([old_batch])
            old_keys = pd.MultiIndex.from_arrays([
                old_table["symbol"].to_pandas(), pd.to_datetime(old_table["date"].to_pandas())
            ], names=["symbol", "date"])
            locations = v2_index.get_indexer(old_keys)
            if np.any(locations < 0):
                raise RuntimeError("正式库存在V2中找不到的主键")
            if used_v2_rows[locations].any():
                raise RuntimeError("正式库存在重复主键")
            used_v2_rows[locations] = True
            for target, source in REPLACEMENTS.items():
                index = old_table.schema.get_field_index(target)
                values = pa.array(v2_frame[source].to_numpy()[locations], type=old_table.schema.field(index).type)
                old_table = old_table.set_column(index, target, values)
            if writer is None:
                writer = pq.ParquetWriter(building, old_table.schema, compression="zstd", compression_level=3)
            writer.write_table(old_table)
            rows_written += old_table.num_rows
        if writer is not None:
            writer.close(); writer = None
        old_pf.close(); old_pf = None
        if not used_v2_rows.all():
            raise RuntimeError("V2存在未写入正式库的主键")
        rebuilt = pq.ParquetFile(building)
        original = pq.ParquetFile(backup)
        if rows_written != original.metadata.num_rows or rebuilt.metadata.num_rows != original.metadata.num_rows:
            raise RuntimeError("重建后行数校验失败")
        if rebuilt.schema_arrow.names != original.schema_arrow.names:
            raise RuntimeError("重建后列顺序校验失败")
        rebuilt.close(); original.close()
        os.replace(building, official)
    except BaseException:
        if writer is not None:
            writer.close()
        if old_pf is not None:
            old_pf.close()
        if building.exists():
            building.unlink()
        if not official.exists() and backup.exists():
            os.replace(backup, official)
        raise

    audit = {
        "timestamp": timestamp, "official": str(official), "backup": str(backup),
        "rows": rows_written, "replaced_columns": REPLACEMENTS,
        "max_dd_columns_replaced": False, "backup_recoverable": True,
    }
    report = REPORTS_DIR / "factor_validation" / "reversal_momentum_v2" / "apply_audit.json"
    report.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
