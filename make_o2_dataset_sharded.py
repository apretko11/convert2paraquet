# make_o2_dataset_sharded.py
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path("compiled_results/compiled_output")
OUTDIR = Path("hf_o2_sharded")
ROWS_PER_SHARD = 60000  # ~100MB shards given your earlier avg row size
COMPRESSION = "snappy"  # good default

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_shard(shard_idx: int, rows: list[dict]):
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        pa.field("x86", pa.string()),
        pa.field("llvm", pa.string()),
        pa.field("file", pa.string()),
    ]))
    shard_path = OUTDIR / f"train-{shard_idx:05d}.parquet"
    pq.write_table(table, shard_path, compression=COMPRESSION)
    print(f"Wrote {shard_path} ({table.num_rows} rows)")

def main():
    # Fresh output dir
    if OUTDIR.exists():
        print(f"Removing old folder: {OUTDIR}")
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Pre-list .ll files so tqdm can show a total
    ll_files = list(ROOT.rglob("*.ll"))
    print(f"Found {len(ll_files):,} .ll files; pairing with .o2.clang.x86.s ...")

    buffer: list[dict] = []
    shard_idx = 0
    total_pairs = 0

    for ll_path in tqdm(ll_files, desc="Pairing (.ll ↔ .o2.s)"):
        base = ll_path.with_suffix("")  # drop .ll
        asm_path = Path(str(base) + ".o2.clang.x86.s")
        if not asm_path.exists():
            continue

        rel_base = ll_path.relative_to(ROOT).with_suffix("")
        buffer.append({
            "x86": read_text(asm_path),
            "llvm": read_text(ll_path),
            "file": str(rel_base),
        })
        total_pairs += 1

        if len(buffer) >= ROWS_PER_SHARD:
            write_shard(shard_idx, buffer)
            shard_idx += 1
            buffer.clear()

    # Final tail shard
    write_shard(shard_idx, buffer)
    print(f"Done. Total O2 pairs: {total_pairs:,}. Shards dir: {OUTDIR}")

if __name__ == "__main__":
    main()

