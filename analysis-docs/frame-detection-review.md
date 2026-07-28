# Detection-Based Surgical Foreign-Object Review

Review date: 2026-07-26

## Research question

Can an object detector or instance-segmentation model provide reliable
grounding and exact counts for the FOCUS Frame track, especially for the small
and repeated objects that the current VLM undercounts?

The short answer is **yes, detection is well supported as a direction**, but
the published evidence is strongest for single-class gauze detection. Exact
multi-class counting of deployed clips, needles, sponges, drains, specimen
bags, specimens, and gallstones in real laparoscopic frames remains
under-studied.

## Metric warning

Results below are not directly rankable:

- Some papers report mAP at IoU 0.5, while others report stricter COCO
  mAP@[0.5:0.95] or do not state the IoU threshold clearly.
- Simulator, clinical, tabletop, radiograph, and ultrasound data have very
  different difficulty.
- A random frame split can place adjacent frames from the same video in train
  and test, whereas a patient/video-disjoint split tests genuine
  generalisation.
- Detection mAP or presence accuracy is not exact count accuracy.

## Most relevant laparoscopic studies

| Study | Data scale and setting | Method | Reported performance | Relevance and limitation |
| --- | --- | --- | --- | --- |
| [de la Fuente López et al., 2020 — Automatic gauze tracking in laparoscopic surgery](https://uvadoc.uva.es/bitstream/handle/10324/65756/Automatic%20gauze%20tracking%20in%20laparoscopic%20surgery%20using%20image%20texture.pdf?sequence=1) | 6,673 image patches: 1,782 gauze and 4,891 background; laparoscopic simulator with animal organs | ResNet-50 patch classifier | Precision 1.00 and sensitivity 0.97 overall; stained-gauze recall 0.93 | Early evidence that deep features recognise gauze, but it is simulated and classifies patches rather than complete instances. |
| [Sánchez-Brizuela et al., 2022 — Gauze detection and segmentation in minimally invasive surgery video](https://www.mdpi.com/1424-8220/22/14/5180) | 42 simulator videos: 30 with gauze and 12 without; 4,003 manually segmented frames sampled from 18 positive videos | YOLOv3 and U-Net | YOLO: precision 94.34%, recall 76.00%, F1 84.18%, reported mAP 74.61%, 34.94 FPS. U-Net IoU approximately 0.85 at more than 30 FPS | Demonstrates real-time feasibility, but 76% recall is insufficient for exact multi-instance counting. |
| [Lai et al., 2023 — Intraoperative detection of surgical gauze](https://link.springer.com/article/10.1007/s10439-022-03033-9), [methods preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4037955) | 3,000 real human laparoscopic colorectal images: 1,250 train and 1,750 test | YOLOv5x6 | Precision 0.920, recall 0.828, reported mAP 0.881, 11.3 ms/image | Directly relevant clinical imagery, but only one gauze class; the accessible description does not establish a strict video-disjoint split. |
| [Ao et al., 2025 — Multi-centre gauze detection in laparoscopic liver surgery](https://link.springer.com/article/10.1186/s40001-025-03190-2) | 33 real surgical videos from two hospitals; approximately 1,616 annotated frames: 1,267 train, 124 internal test, and 225 external test in the methods | YOLOv8n, SSD, Faster R-CNN; FCN-ResNet101 segmentation | YOLOv8n internal: P 0.903, R 0.844, mAP50 0.914. External: P 0.910, R 0.829, mAP50 0.906. External segmentation Dice 0.908 | Strong cross-hospital evidence. It is still one-class gauze detection, and the paper contains a small inconsistency in external-frame totals. |
| [Zhou et al., 2026 — YOLOv11m surgical-object instance segmentation](https://www.zgsyz.com/zgsywk/EN/10.19538/j.cjps.issn1005-2208.2026.03.14) | 3,180 real laparoscopic images; 16,239/4,022/2,196 annotated targets in train/validation/test | YOLOv11m instance segmentation | Overall box and mask mAP50 0.908; box P 0.848 and R 0.884. Gauze mAP50 above 0.90; biological clips and needles between 0.80 and 0.90 | The closest class coverage to FOCUS. The image-level random split is not clearly patient/video-disjoint, so adjacent-frame leakage is possible. Exact count accuracy is not reported. |
| [Yoon et al., 2021 — hSDB-instrument](https://arxiv.org/abs/2110.12555), [official dataset/results](https://hsdb-instrument.github.io/) | 48 clinical cases: 26,919 real cholecystectomy frames and 42,891 real gastrectomy frames, plus 25,675 synthetic frames | Cascade R-CNN, FoveaBox, and related detectors | Best listed strict COCO mAP@[0.5:0.95] is approximately 27.1 for cholecystectomy and 40.7 for gastrectomy | A much larger, multi-class, case-disjoint clinical benchmark. It includes needles, specimen bags, and tubes, but not the complete deployed-object FOCUS ontology. The lower strict mAP illustrates the difficulty hidden by one-class mAP50 results. |

## Supporting studies in easier or different settings

| Study | Data scale | Method and performance | Why it should not be treated as a laparoscopic expectation |
| --- | --- | --- | --- |
| [Deol et al., 2024 — Automated surgical instrument detection and counting](https://pmc.ncbi.nlm.nih.gov/articles/PMC11265075/) | 1,004 controlled operating-table images, 13,213 instances, 11 classes; 603/201/200 train/validation/test images | YOLOv9: P 98.5%, R 99.9%, mAP50 99.4%, mAP50–95 88.4%; 40.4 FPS on V100 | Clean tabletop views, stable scale, little tissue-like background, and no intra-abdominal occlusion. It proves counting can be deterministic when instances are visually separable. |
| [Abramson et al., 2022 — Ultrasound cotton-ball foreign-body detection](https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2022.1040066/full) | 7,121 ultrasound images from 10 ex-vivo porcine brains: 4,898/1,046/1,057 train/validation/test | VGG16-based network: mean IoU 0.92 and reported sensitivity, specificity, and accuracy around 0.99 | Different modality and ex-vivo setting. Only two human cases were explored, and true-negative human cases failed. |
| [Wang et al., 2025 — Hopkins RFOs Bench](https://arxiv.org/pdf/2507.06937) | 444 chest radiographs: 144 critical retained-foreign-object cases, 150 negative cases, and 150 noncritical cases; patient-disjoint 70/10/20 split | Base YOLO: accuracy 73.8%, FNR 0.33, AUC 0.62, FROC 51.2. Adding 2,000 physics-based synthetic images: accuracy 78.7%, FNR 0.21, AUC 0.77, FROC 59.7 | Radiography rather than laparoscopic RGB. Its useful lesson is that physically plausible synthetic data helped, whereas less constrained synthetic generation could degrade performance. |

## Evidence synthesis

### What has been demonstrated

For one-class gauze detection in real laparoscopic images, the strongest
studies report roughly:

- Precision: 0.90–0.92
- Recall: 0.83–0.84
- Reported mAP or mAP50: 0.88–0.91
- Annotation scale: approximately 1,000–3,000 clinical frames

This is sufficient for presence detection and useful visual grounding. It is
not automatically sufficient for exact counting.

### Why ordinary detector recall is not enough

Suppose each of \(n\) objects is detected independently with recall \(r\), and
ignore false positives. The probability that all objects are found is
approximately:

\[
P(\text{all found}) \approx r^n
\]

At the approximately 0.83 recall reported by several gauze studies:

- Five objects: \(0.83^5 \approx 39\%\)
- Seven objects: \(0.83^7 \approx 27\%\)

To reach an 80% all-detected probability at seven objects under this simplified
assumption requires per-instance recall around 0.969. False positives,
duplicate detections, and correlated occlusion make the real requirement even
harder.

This is illustrative reasoning, not a result reported by the cited studies.
It explains why a detector can have respectable mAP while remaining poor at
FOCUS exact counts.

### The main research gap

In the literature reviewed here, few studies report:

- Exact count accuracy per frame
- Count MAE and signed undercount bias
- Accuracy or recall stratified by the true object count
- Performance on clusters of small deployed clips
- Multi-class counting across the complete FOCUS ontology
- Cross-hospital, video-disjoint evaluation for all classes
- Downstream VQA improvement from detector evidence

This gap is a credible research contribution rather than merely an engineering
task.

## Connection to the current Frame result

The current VLM's failures match the gap in the literature:

| Current Frame task | Accuracy |
| --- | ---: |
| Distinct foreign-object classes | 75.2% |
| All visible object instances | 39.6% |
| Clip count | 30.8% |
| Counts of six or more | 1.0% |
| Sponge count | 92.8% |

The model can identify categories and count large isolated objects, but it
does not reliably separate repeated small instances. A detector or
instance-segmentation model directly supervises the missing operation:
localising every instance.

The stale multi-label output adapter documented in
[frame-improve-v1.md](frame-improve-v1.md) must be fixed first. Detection should
target the remaining visual weakness rather than compensate for a deterministic
formatting bug.

## Recommended experimental programme

### 1. Define the task around exact counting

Use the complete FOCUS class vocabulary and annotate every visible instance,
not only the object mentioned in the associated question. At minimum, each
annotation should contain:

- Class
- Bounding box or centre point
- Occlusion flag
- Truncation/visibility flag
- Source dataset, video, and timestamp

Instance masks are particularly valuable for touching or overlapping clips
and for distinguishing anatomy-caused occlusion. If full masks are too
expensive, boxes plus centre points are a practical first stage.

### 2. Start with a learning-curve pilot

A reasonable first sequence is:

1. Annotate 1,000 unique frames to establish feasibility.
2. Expand to 2,000 frames if small-object recall is still improving.
3. Target 3,000–5,000 frames for a serious multi-class model, with more data
   for rare classes if required.

The one-class literature suggests that 1,000–4,000 images can produce a strong
baseline. Multi-class FOCUS detection is harder, so image count alone is not
enough. The pilot must be stratified by:

- HeiCo and LapChole
- Object class
- Count ranges: 1, 2–3, 4–5, and 6+
- Clip clusters and other small objects
- Blur, blood, smoke, glare, occlusion, and partial visibility
- Negative frames and confusing instruments/anatomy

Use video- or patient-disjoint train, validation, and test splits. Random frame
splits risk measuring memorisation of adjacent frames.

### 3. Establish several model families

Recommended baselines:

- **YOLOv11 or YOLOv8 at high input resolution:** fast, simple, and directly
  comparable with recent gauze work.
- **RT-DETR or Faster R-CNN:** useful independent detector family and a check
  against YOLO-specific NMS behaviour.
- **Instance segmentation:** YOLO-seg or Mask R-CNN for overlapping clips and
  object-state/occlusion analysis.
- **Point or density-map detection:** worth testing if clip boxes are
  ill-defined but clip centres remain annotatable.

For clips, use high-resolution training, crop/tile augmentation, small-object
sampling, and carefully tuned NMS. Standard NMS can suppress two adjacent
clips as if they were duplicate boxes.

### 4. Use counting-specific metrics

Report conventional detection metrics, but do not optimise only for mAP:

- AP50 and AP50–95 by class
- Average recall and recall by object size
- Recall stratified by true object count
- Exact per-frame count accuracy
- Per-class and overall count MAE
- Signed count bias and undercount rate
- Presence/co-occurrence accuracy
- Cross-dataset and per-video performance

The primary model-selection metric for this project should combine exact count
accuracy with small-object recall. A detector with higher mAP but lower
high-count recall may be worse for FOCUS.

### 5. Integrate detection with the VLM

The first integration should remain modular:

1. Run the detector once for the frame.
2. Convert detections to structured evidence:
   `class, confidence, centre, quadrant, box/mask, total count`.
3. Answer number, binary presence, co-occurrence, and coarse quadrant questions
   deterministically when the question can be parsed safely.
4. Provide the same structured evidence to the VLM for open-ended, state, and
   anatomical questions.
5. Fall back to the VLM when detector confidence or question parsing is
   ambiguous.

This makes improvements attributable and debuggable. End-to-end auxiliary
grounding losses can be explored later, after the detector establishes that
the required instances are recoverable from the pixels.

## Overall opinion

A detection branch is one of the most justified improvements for the Frame
track. Published work shows that surgical foreign objects can be detected at
useful speed and accuracy, including on clinical data and external hospitals.
However, the current literature's usual recall around 0.83 is not enough for
high-count exact answers, and near-perfect tabletop results are not a realistic
target.

The strongest research direction is therefore not simply “add YOLO.” It is:

> High-recall, high-resolution, multi-class instance grounding evaluated
> explicitly for exact surgical-object counting, followed by structured
> detector–VLM fusion.

This directly targets the current model's 30.8% clip-count accuracy and 1.0%
accuracy on counts of six or more, while also creating a publishable evaluation
setting that is largely absent from existing work.
