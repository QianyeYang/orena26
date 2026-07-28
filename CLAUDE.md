When writing to this doc, using breif words but do not miss any critical information.

## Overview
This repo is for doing the challenges of Foreign Object Contextual Understanding in Surgery

It has 3 tracks:
https://frame.orena-focus-challenge.org/
https://segment.orena-focus-challenge.org/
https://procedure.orena-focus-challenge.org/

The tasks are buliding VLM pipelines for VQA tasks. 

The environment, ALWAYS using conda env "orena", don't mixup with other envs 

## Directories
    ./data: storage of the data
    ./src: shared source code, utilised by all methods
    ./docs: documentations, guidellines, important records, md files
    ./scripts: utility scripts
    ./track-xxxx: separate folders for specific track challenges.
    ./submissions: place to generate submission dockers
    ./tmp: temporary folder for one-time use scripts, code, docs, any materials
    ./os-models: stores the opensource VLMs and LLMs
    ./result-summary/: stores track- and experiment-specific numerical reports with dataset and capability distributions; keep the indexes clean and organised.
    ./result-summary.md: historical cross-track snapshot retained for provenance.

And for each of the ./track-xxxx, the sub directories are orgainsed as follows:
    
    each method group should be in the seperate folder, due to large architecture difference, e,g, ./track-xxxx/baseline/
    ./<method>/architecture.md : explain the architecture of the method 
    ./<method>/logs: train/inference logs, and also the place to same the model ckpts
    ./<method>/src: ONLY method specific code, for sharing code, put it in the upper-level src folder, rather than here
    ./<method>/scripts: scripts that used to submit to SLURM, for training and testing 
    

## Data
Videos:
- HeiCo (30 files, 149.6 GiB): ./data/focus/heico/videos/
- LapChole (170 files, 90.3 GiB): ./data/focus/lapchole/videos/

VQA annotations:
- HeiCo: ./data/parquet/{frame,segment,procedure}/{train,test}/0000.parquet
- LapChole: ./data/parquet/lapchole/{frame,segment,procedure}/{train,test}/0000.parquet
- RULE: always load annotations with pd.read_parquet() directly from ./data/parquet/, NOT via load_dataset() — parquet is ~160x faster (4ms vs 650ms per load)
- HeiCo rows: frame 8000/4000, segment 8000/4000, procedure 4000/2000 (train/test)
- LapChole rows: frame 5748/2252, segment 5746/2254, procedure 2873/1127 (train/test)
