from pathlib import Path
from datasets import Dataset
from tqdm import tqdm  # <-- add this

ROOT = Path("compiled_results/compiled_output")

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

examples = []

# Pre-list .ll files so tqdm can show total
ll_files = list(ROOT.rglob("*.ll"))

for ll_path in tqdm(ll_files, desc="Scanning & pairing (.ll ↔ .o2.s)"):
    base = ll_path.with_suffix("")  # remove .ll
    asm_path = Path(str(base) + ".o2.clang.x86.s")  # O2 suffix
    if asm_path.exists():
        rel_base = ll_path.relative_to(ROOT).with_suffix("")
        examples.append({
            "x86": read_text(asm_path),
            "llvm": read_text(ll_path),
            "file": str(rel_base)
        })

print(f"Collected {len(examples)} O2 pairs")

ds = Dataset.from_list(examples)
print("Writing parquet: hf_parquet_out_o2.parquet …")
ds.to_parquet("hf_parquet_out_o2.parquet")
print("Done. Nap time")
