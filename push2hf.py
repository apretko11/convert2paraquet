from datasets import Dataset

# Load from the shards we just created
ds = Dataset.from_parquet("hf_o0_sharded/*.parquet")

# Push to the Hub (will create repo if it doesn't exist)
ds.push_to_hub("adpretko/x86-to-llvm-o0", private=True)
