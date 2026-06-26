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
    ./result-summary.md: stores the techinical numerical results of each methods & baselines, ALWAYS need to keep it clean and well organised, rather than just appending new results in it.

And for each of the ./track-xxxx, the sub directories are orgainsed as follows:
    
    each method group should be in the seperate folder, due to large architecture difference, e,g, ./track-xxxx/baseline/
    ./<method>/architecture.md : explain the architecture of the method 
    ./<method>/logs: train/inference logs, and also the place to same the model ckpts
    ./<method>/src: ONLY method specific code, for sharing code, put it in the upper-level src folder, rather than here
    ./<method>/scripts: scripts that used to submit to SLURM, for training and testing 
    

## Data
Videos (161 GB, 30 .avi files): ./data/focus/heico/videos/

VQA annotations (parquet): ./data/parquet/{frame,segment,procedure}/{train,test}/0000.parquet
- RULE: always load annotations with pd.read_parquet() directly from ./data/parquet/, NOT via load_dataset() — parquet is ~160x faster (4ms vs 650ms per load)
- Row counts: frame 4000/2000, segment 4000/2000, procedure 2000/1000 (train/test)
- Content verified identical to HF Arrow cache