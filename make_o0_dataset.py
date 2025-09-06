import os
from pathlib import Path
from datasets import Dataset

ROOT = Path("compiled_results/compiled_output")

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

examples = []
for ll_path in ROOT.rglob("*.ll"):
    base = ll_path.with_suffix("")  # remove .ll
    asm_path = Path(str(base) + ".o0.clang.x86.s")
    if asm_path.exists():
        rel_base = ll_path.relative_to(ROOT).with_suffix("")
        examples.append({
            "x86": read_text(asm_path),
            "llvm": read_text(ll_path),
            "file": str(rel_base)
        })

print(f"Collected {len(examples)} O0 pairs")

ds = Dataset.from_list(examples)
ds.to_parquet("hf_parquet_out")
