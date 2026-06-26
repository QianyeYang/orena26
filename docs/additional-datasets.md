# Existing Surgical Video Datasets Related to ORena FOCUS

## 1. Background

The ORena FOCUS Challenge is centered on Foreign Object Contextual Understanding in Surgery. Unlike conventional surgical video benchmarks that mainly focus on surgical phase recognition, tool detection, instrument segmentation, or action recognition, FOCUS targets clinically motivated questions about foreign objects such as sponges, clips, needles, silicone loops, drains, or specimen bags.

The key difference is that FOCUS is not merely asking whether an object is visible in a frame. It asks models to understand the contextual status of foreign objects: whether they are visible, manipulated, occluded, inserted, removed, counted, or still present during a surgical procedure. This makes the challenge closer to surgical safety and retained foreign object prevention than traditional surgical scene analysis.

The released HeiCo-FOCUS VQA dataset is built on Heidelberg colorectal surgery videos and contains long-context surgical VQA annotations. The dataset requires models to track multiple foreign objects as they are inserted, manipulated, occluded, and removed during procedures that may last several hours.

---

## 2. Dataset Comparison

| Dataset | Surgical domain | Main annotations | Relation to FOCUS | Key limitation for FOCUS |
|---|---|---|---|---|
| HeiCo / ROBUST-MIS | Laparoscopic colorectal surgery | Surgical phase labels, instrument presence, instrument instance segmentation masks | Strongly related because HeiCo-FOCUS is built on HeiCo videos | Original HeiCo focuses on surgical instruments and phases, not foreign-object VQA or retrieval-status reasoning |
| Cholec80 | Laparoscopic cholecystectomy | Surgical phases and tool presence annotations | Useful for learning surgical visual context, tool presence, and procedural phase priors | Does not focus on foreign objects such as sponges or needles; no VQA; no retrieval-status reasoning |
| m2cai16-tool / m2cai16-tool-locations | Laparoscopic cholecystectomy | Tool presence and tool spatial annotations | Useful for training visual grounding or tool localization modules | Mainly instrument-focused; not designed for foreign-object tracking or surgical safety QA |
| CholecT50 / CholecTriplet | Laparoscopic cholecystectomy | Surgical action triplets in the form <instrument, verb, target>; phase labels | Relevant to interaction and event understanding, especially for SEGMENT-style reasoning | Focuses on instrument-action-target recognition, not foreign-object insertion/removal/accounting |
| CholecTrack20 | Laparoscopic cholecystectomy | Multi-class surgical tool tracking, tool identity, spatial location, phase, visual conditions | Useful for object identity matching and tracking ideas | Tracks surgical tools, not foreign objects requiring retrieval or accounting |
| Surgical-VQA / EndoVis18-VQA / PitVQA | Endoscopic / laparoscopic surgical scenes | Image-based surgical visual question answering | Related in task format because they are surgical VQA datasets | Mostly static or short-context VQA; not centered on long-horizon foreign-object safety |
| SSG-VQA / SSG-QA | Surgical scene understanding | Scene-graph-based VQA generated from instruments, anatomy, spatial relations, and actions | Highly relevant for structured reasoning and geometric grounding | Does not specifically model foreign-object status, insertion/removal, or retained-object safety |
| Recent surgical VideoQA datasets, e.g. SurgViVQA / CholeVidQA-style datasets | Surgical video understanding | Temporally grounded surgical video QA, tool-tissue interaction, procedural assessment | Relevant to temporal VQA and video reasoning | Usually focus on surgical actions, anatomy, difficulty, or assessment rather than foreign-object retrieval status |

---

## 3. Dataset Notes


### 3.1 HeiCo / ROBUST-MIS

**Download / project page:** [HeiCo on Synapse](https://www.synapse.org/Synapse:syn21903917) / [ROBUST-MIS challenge data on Synapse](https://www.synapse.org/Synapse:syn18779624)

HeiCo is one of the most directly related datasets because the first released HeiCo-FOCUS VQA data is built on Heidelberg colorectal surgical videos. The original HeiCo dataset contains 30 laparoscopic colorectal videos and includes surgical phase labels, instrument presence annotations, and instance-wise instrument segmentation masks for more than 10,000 frames.

However, the original HeiCo annotations are designed for instrument detection, segmentation, and phase recognition. They are not designed for foreign-object contextual understanding. HeiCo-FOCUS extends this video source into a new VQA task focused on foreign objects and long-context reasoning.

For FOCUS, HeiCo is useful because it provides the surgical visual domain and the underlying videos, but the original dense labels are not directly aligned with foreign-object safety questions.

---


### 3.2 Cholec80

**Download / project page:** [CAMMA datasets page](https://camma.unistra.fr/datasets/) / [TF-Cholec80 helper repository](https://github.com/CAMMA-public/TF-Cholec80)

Cholec80 is a widely used public laparoscopic cholecystectomy dataset. It contains 80 cholecystectomy videos with surgical phase labels and tool presence annotations. It is useful for learning general laparoscopic visual features, tool presence, and procedural context.

The overlap with FOCUS is indirect. Cholec80 includes surgical tools and some objects that may appear relevant, such as specimen bags or clip-related tools, but it does not provide annotations for foreign-object accounting, insertion/removal status, or long-horizon object memory.

Therefore, Cholec80 is useful for pretraining or domain adaptation, but it cannot directly solve the main FOCUS task.

---


### 3.3 m2cai16-tool and m2cai16-tool-locations

**Download / project page:** [CAMMA datasets page](https://camma.unistra.fr/datasets/) / [m2cai16-tool-locations on OpenDataLab](https://opendatalab.com/OpenDataLab/m2cai16-tool-locations/download)

The m2cai16 tool datasets are useful for surgical tool detection and localization. The tool-location version provides spatial annotations for surgical instruments in laparoscopic cholecystectomy videos.

This type of data is relevant to FOCUS because many FRAME questions require object localization or spatial reasoning. However, m2cai16 is still mainly about surgical instruments, not foreign objects such as sponges, needles, loops, or drains. It also does not provide VQA annotations or long-term retrieval-status labels.

Thus, m2cai16 can help train a visual grounding module, but it is not a foreign-object contextual understanding dataset.

---


### 3.4 CholecT50 / CholecTriplet

**Download / project page:** [CholecT50 GitHub repository](https://github.com/CAMMA-public/cholect50) / [CholecTriplet 2022 data page](https://cholectriplet2022.grand-challenge.org/data/) / [CAMMA datasets page](https://camma.unistra.fr/datasets/)

CholecT50 provides fine-grained surgical action triplets in the form:

text <instrument, verb, target> 

This is relevant to FOCUS because SEGMENT and PROCEDURE questions may involve manipulation events, such as whether an object is grasped, moved, inserted, or removed. CholecT50 can therefore help models learn surgical action and interaction patterns.

However, the semantic focus is different. CholecT50 is about instrument-action-target recognition, while FOCUS is about foreign-object status and safety. CholecT50 does not tell us whether a sponge or needle has been introduced, removed, or remains in the body cavity.

---


### 3.5 CholecTrack20

**Download / project page:** [CholecTrack20 GitHub repository](https://github.com/CAMMA-public/cholectrack20) / [CAMMA datasets page](https://camma.unistra.fr/datasets/)

CholecTrack20 is relevant because it provides surgical tool tracking annotations, including object identity and temporal consistency. This is conceptually useful for FOCUS, especially for SEGMENT and PROCEDURE tracks, where models need to maintain object identity over time.

However, CholecTrack20 tracks surgical tools rather than foreign objects that require accounting. It is useful for designing tracking-based architectures, but it does not provide the specific supervision needed for retained foreign-object reasoning.

---


### 3.6 Surgical-VQA / EndoVis18-VQA / PitVQA

**Download / project page:** [Surgical-VQA GitHub repository](https://github.com/lalithjets/Surgical_VQA) / [PitVQA GitHub repository](https://github.com/mobarakol/PitVQA) / [PitVQA dataset record at UCL Research Data Repository](https://rdr.ucl.ac.uk/articles/dataset/PitVQA_A_Dataset_of_Visual_Question_Answering_in_Pituitary_Surgery/27004666)

These datasets are important because they show that surgical VQA existed before FOCUS. They typically ask questions about surgical scenes, instruments, anatomy, or procedural context.

Therefore, FOCUS should not be described as the first public surgical VQA dataset. That would be inaccurate.

The difference is that previous surgical VQA datasets are usually image-based or short-context, while FOCUS is centered on foreign-object contextual understanding and long-horizon reasoning.

---


### 3.7 SSG-VQA / SSG-QA

**Download / project page:** [SSG-VQA GitHub repository](https://github.com/CAMMA-public/SSG-VQA)

SSG-VQA is particularly relevant from a methodological perspective. It builds surgical scene graphs using spatial and action information from instruments and anatomy, then generates VQA pairs from these structured representations.

This is important for FOCUS because many FOCUS questions implicitly require scene-graph-like reasoning:

text Which object is in the top/right region? Is the foreign object grasped by an instrument? Which object is partially occluded? How many foreign object instances are visible? 

The limitation is that SSG-VQA does not specifically focus on foreign-object accounting, insertion/removal events, or retrieval-status reasoning.

---


### 3.8 SurgViVQA / REAL-Colon-VQA

**Download / project page:** [SurgViVQA GitHub repository](https://github.com/madratak/SurgViVQA)

SurgViVQA and REAL-Colon-VQA are relevant because they explicitly move surgical VQA from isolated images toward temporally grounded video question answering. They are useful references for temporal surgical VQA, but their focus is still broader surgical video understanding rather than foreign-object accounting or retrieval-status reasoning.

---

## 4. Is HeiCo-FOCUS the First Public Foreign Object Dataset?

The safest conclusion is:

HeiCo-FOCUS is not the first public surgical video dataset.  
Many public surgical video datasets already exist, including HeiCo, Cholec80, m2cai16, CholecT50, and others.

HeiCo-FOCUS is also not the first public surgical VQA dataset.  
Surgical-VQA, EndoVis18-VQA, PitVQA, and SSG-VQA already introduced surgical VQA-style tasks.

However, HeiCo-FOCUS appears to be one of the first, and likely the first publicly released benchmark, specifically centered on:

text foreign-object contextual understanding long-horizon foreign-object tracking foreign-object insertion/manipulation/removal reasoning retrieval-status reasoning retained foreign object safety 

A careful way to write this is:

> To the best of our knowledge, HeiCo-FOCUS is the first publicly released surgical VQA benchmark centered on foreign-object contextual understanding and long-horizon retrieval-status reasoning in minimally invasive surgery.

This statement is more accurate than saying it is the first surgical video dataset or the first surgical VQA dataset.

---

## 5. Key Insight for Model Design

Existing public datasets can help with several sub-problems:

| Sub-problem | Useful external datasets |
|---|---|
| Surgical visual representation learning | Cholec80, HeiCo, m2cai16 |
| Tool detection and localization | m2cai16-tool-location, HeiCo, CholecInstanceSeg-style datasets |
| Surgical phase recognition | Cholec80, HeiCo |
| Instrument-action understanding | CholecT50 / CholecTriplet |
| Tool tracking and identity matching | CholecTrack20 |
| Surgical VQA formatting and language supervision | Surgical-VQA, PitVQA, EndoVis18-VQA, SSG-VQA |

But these datasets do not directly provide supervision for the central FOCUS problem:

text foreign object identity foreign object count foreign object location foreign object manipulation foreign object insertion/removal foreign object retrieval status long-term object memory 

Therefore, the main value of FOCUS is not dense visual annotation. Its value is that it provides weak but clinically meaningful VQA supervision for a task that previous surgical datasets do not directly address.

---

## 6. Practical Implication

For building a model, external datasets should not be treated as direct replacements for FOCUS. Instead, they should be used as auxiliary resources:

1. Use Cholec80 / HeiCo / m2cai16 to learn surgical visual representations.
2. Use CholecT50 or CholecTriplet to learn surgical interaction and action concepts.
3. Use CholecTrack20-like tracking data to design object identity and temporal memory modules.
4. Use Surgical-VQA or SSG-VQA to pretrain question answering and structured reasoning.
5. Use FOCUS itself to learn the actual foreign-object question taxonomy, answer formats, and retrieval-status reasoning.

The most important modeling direction is therefore not a pure VLM baseline, but a structured system that converts video into a foreign-object memory representation and answers questions by querying that memory.

A possible representation is:

text video/frame     -> foreign object candidates     -> object class, location, visibility, interaction state     -> temporal tracklets     -> insertion/removal/manipulation events     -> object memory     -> question-conditioned retrieval     -> short normalized answer 

This design is better aligned with FOCUS than simply feeding sampled frames into a general-purpose VLM.