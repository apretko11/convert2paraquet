from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("compiled_results/compiled_output")
OUT = "hf_o0_dataset.parquet"

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

# define schema
schema = pa.schema([
    ("x86", pa.string()),
    ("llvm", pa.string()),
    ("file", pa.string()),
])

writer = pq.ParquetWriter(OUT, schema)

batch_size = 1000
buffer = []

count = 0
for ll_path in ROOT.rglob("*.ll"):
    asm_path = Path(str(ll_path.with_suffix("")) + ".o0.clang.x86.s")
    if not asm_path.exists():
        continue

    rel_base = ll_path.relative_to(ROOT).with_suffix("")
    buffer.append({
        "x86": read_text(asm_path),
        "llvm": read_text(ll_path),
        "file": str(rel_base),
    })

    if len(buffer) >= batch_size:
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table)
        count += len(buffer)
        print(f"Written {count} rows...", flush=True)
        buffer = []

# write any leftovers
if buffer:
    table = pa.Table.from_pylist(buffer, schema=schema)
    writer.write_table(table)
    count += len(buffer)

writer.close()
print(f"✅ Finished. Total rows written: {count}")
