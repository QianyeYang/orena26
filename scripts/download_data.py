from focus import FocusConfig, set_config, download
from datasets import load_dataset
import os

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "focus")
data_dir = os.path.abspath(data_dir)
os.makedirs(data_dir, exist_ok=True)

set_config(FocusConfig(root_dir=data_dir))

print(f"Downloading HeiCo videos to {data_dir} ...")
download("heico")
print("Video download complete.")

print("Downloading VQA annotations from Hugging Face ...")
for track in ["frame", "segment", "procedure"]:
    print(f"  Loading {track} ...")
    ds = load_dataset("orena-dkfz/heico-focus-vqa", track, split="train")
    print(f"  {track}: {len(ds)} samples")

print("All downloads complete.")
