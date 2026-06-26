import os
import time
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset

PARQUET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")
PARQUET_DIR = os.path.abspath(PARQUET_DIR)

TRACKS = ["frame", "segment", "procedure"]
SPLITS = ["train", "test"]

# Download parquet files
print("=== Downloading parquet files ===")
for track in TRACKS:
    for split in SPLITS:
        dest_dir = os.path.join(PARQUET_DIR, track, split)
        os.makedirs(dest_dir, exist_ok=True)
        path = f"{track}/{split}/0000.parquet"
        local = hf_hub_download(
            repo_id="orena-dkfz/heico-focus-vqa",
            filename=path,
            repo_type="dataset",
            revision="refs/convert/parquet",
            local_dir=PARQUET_DIR,
        )
        print(f"  {path} -> {local}")

print("\n=== Content check (row counts) ===")
for track in TRACKS:
    for split in SPLITS:
        pq_path = os.path.join(PARQUET_DIR, track, split, "0000.parquet")
        df_pq = pd.read_parquet(pq_path)
        ds = load_dataset("orena-dkfz/heico-focus-vqa", track, split=split)
        print(f"  {track}/{split}: parquet={len(df_pq)} rows, arrow cache={len(ds)} rows, match={len(df_pq)==len(ds)}")

print("\n=== Loading speed comparison ===")
for track in TRACKS:
    # Parquet
    pq_path = os.path.join(PARQUET_DIR, track, "train", "0000.parquet")
    t0 = time.perf_counter()
    df_pq = pd.read_parquet(pq_path)
    t_pq = time.perf_counter() - t0

    # Arrow cache via datasets
    t0 = time.perf_counter()
    ds = load_dataset("orena-dkfz/heico-focus-vqa", track, split="train")
    t_arrow = time.perf_counter() - t0

    print(f"  {track}/train: parquet={t_pq:.3f}s  arrow cache={t_arrow:.3f}s")

print("\nDone.")
