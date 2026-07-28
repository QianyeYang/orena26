

# Guideline: Downloading the ORENA FOCUS Dataset

This note explains the simplest way to download and load the dataset for the ORENA FOCUS VQA tasks:

- FRAME: frame-level visual question answering
- SEGMENT: segment-level visual question answering
- PROCEDURE: procedure-level visual question answering

## 1. Install the package

Create the repository environment:

```bash
conda env create -f environment.yml
conda activate orena
```

## 2. Choose a local data directory

Decide where the videos should be stored. For example:

```text
./data/focus
```

This directory contains one folder per release:

```text
data/focus/heico/
data/focus/lapchole/
```

## 3. Download the HeiCo videos

Use the official `focus` package instead of manually downloading files from the challenge websites.

```python
from focus import FocusConfig, set_config, download

set_config(FocusConfig(root_dir="data/focus"))
download("heico")
```

After downloading, the video files should be stored under:

```text
data/focus/heico/videos/
```

## 4. Download the gated LapChole release

LapChole contains 170 videos (about 90.3 GiB) and requires approved access to
`orena-dkfz/lapchole-focus-vqa`. The repository downloader reads `HF_KEY` from
`~/.bashrc`, with the gitignored `.env` as a fallback.

Check access without downloading:

```bash
conda activate orena
python scripts/download_data.py --datasets lapchole --dry-run
```

Run the large, resumable download as a Slurm job:

```bash
sbatch scripts/download_data.slurm
```

Outputs:

```text
data/focus/lapchole/videos/
data/parquet/lapchole/{frame,segment,procedure}/{train,test}/0000.parquet
logs/download-data-<job-id>.log
```

Re-running the same command resumes or skips completed files.

## 5. Load the VQA annotations

The VQA annotations are hosted on Hugging Face and can be loaded automatically.

For direct Hugging Face loading:

```python
from datasets import load_dataset

frame_ds = load_dataset("orena-dkfz/heico-focus-vqa", "frame", split="train")
segment_ds = load_dataset("orena-dkfz/heico-focus-vqa", "segment", split="train")
procedure_ds = load_dataset("orena-dkfz/heico-focus-vqa", "procedure", split="train")
```

Alternatively, use the `FocusDataset` wrapper:

```python
from focus import FocusConfig, set_config
from focus import FocusDataset, DatasetSplit, Track

set_config(FocusConfig(root_dir="data/focus"))

frame_dataset = FocusDataset("heico", DatasetSplit.TRAIN, Track.FRAME)
request, reference = frame_dataset[0]

print(request.question)
print(reference.answer)
```

## 6. Recommended starting point

Start with the FRAME task first.

FRAME is the easiest task for building an initial VLM pipeline because each sample mainly requires reasoning over a single frame. SEGMENT and PROCEDURE are harder because they require temporal understanding over video clips or full procedures.

A practical development order is:

1. Build and test the pipeline on FRAME.
2. Extend the same code structure to SEGMENT.
3. Finally adapt it to PROCEDURE.

## 7. Minimal pipeline structure

A simple VLM pipeline can follow this structure:

```text
load sample
→ extract frame or video segment
→ send visual input + question to VLM
→ parse model answer
→ compare with reference answer
→ save prediction
```

For the first baseline, avoid complicated video processing. Use FRAME to verify that:

- the dataset can be loaded correctly;
- the video or frame path is valid;
- the VLM receives the correct visual input;
- the output format is compatible with the evaluation script.

## 8. Notes

- The challenge websites describe the tasks, but the easiest dataset access is through the official `orena-focus` package and Hugging Face dataset.
- The Docker submission environment will usually not allow internet access, so all required model weights and data-dependent resources should be prepared before submission.
- Keep local paths configurable rather than hard-coded, especially when moving between a workstation, server, and Docker environment.
