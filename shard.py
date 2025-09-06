import os
import shutil
from datasets import Dataset

# Input + output
infile = "hf_o0_dataset.parquet"
outdir = "hf_o0_sharded"
target_shard_size_mb = 100  # change to 200 if you want ~200MB shards

# Step 1: load dataset
ds = Dataset.from_parquet(infile)
num_rows = len(ds)

# Step 2: compute average row size
file_size_bytes = os.path.getsize(infile)
avg_row_size = file_size_bytes / num_rows
rows_per_shard = int((target_shard_size_mb * 1024 * 1024) / avg_row_size)

print(f"Total rows: {num_rows}")
print(f"File size: {file_size_bytes/1024/1024:.2f} MB")
print(f"Avg row size: {avg_row_size:.2f} bytes")
print(f"Rows per shard (~{target_shard_size_mb}MB): {rows_per_shard}")

# Step 3: prepare output directory
if os.path.exists(outdir):
    print(f"Removing old folder: {outdir}")
    shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)

# Step 4: write shards
for i in range(0, num_rows, rows_per_shard):
    shard = ds.select(range(i, min(i + rows_per_shard, num_rows)))
    shard_file = os.path.join(outdir, f"train-{i//rows_per_shard:05d}.parquet")
    shard.to_parquet(shard_file)
    print(f"✅ Wrote {shard_file} ({len(shard)} rows)")

print("🎉 Done! Shards are in:", outdir)
