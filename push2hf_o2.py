from datasets import Dataset

# Load from the shards we just created
ds = Dataset.from_parquet("hf_o2_sharded/*.parquet")

# Push to the Hub (new dataset ID so it stays separate from O0)
ds.push_to_hub("adpretko/x86-to-llvm-o2", private=True)
