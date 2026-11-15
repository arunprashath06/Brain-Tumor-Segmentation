# Brain Tumor Segmentation and Volumetric Delineation from MRI Scans

**Project report — demo edition**

A deep-learning system that looks at a brain MRI slice and outlines the tumor. It ships as a
Streamlit app where you can browse real patients, run the model, and compare its outline with the
radiologist's ground truth.

---

## 1. What the system does

Input: one axial brain MRI slice (256 × 256, grayscale).
Output: a tumor mask for that slice, plus a probability map and a confidence score.

The app answers four questions for any slice:

1. **Is there a tumor?** — the mask, or an explicit "no tumor predicted".
2. **Where?** — the outline drawn on the scan.
3. **How sure?** — mean and peak probability inside the outline.
4. **How right?** — Dice and IoU against the ground-truth mask, when the dataset is available.

---

## 2. Dataset

| Item | Value |
|---|---|
| Name | Kaggle *LGG MRI Segmentation* (TCGA-LGG) |
| Patients | 110 |
| Image files | 7,858 `.tif` (image + mask pairs) |
| Native resolution | 256 × 256 |
| Labels | binary tumor mask per slice |

Splitting is done by **patient**, never by slice, so slices of the same brain cannot leak between
training and validation.

---

## 3. Models used

All three runs share the same architecture family: a UNet from `segmentation-models-pytorch`, trained
on **2.5D input** — three neighbouring axial slices stacked into **9 channels** — with **scSE
attention** in the decoder.

| Run | Encoder | Folds saved | Epochs | Loss |
|---|---|---|---|---|
| `improved_r34_25d` | ResNet-34 | 0–3 | 80 | Tversky (0.3/0.7) + focal |
| `upgraded_efficientnet_b4` | EfficientNet-B4 | 0 | 100 | Tversky + focal |
| `efficientnet_b4_25d` | EfficientNet-B4 | 0–1 | 100 | Tversky + focal |

The ResNet-34 checkpoint has **24.6 M parameters**. The app rebuilds each architecture from its
`config.json`, reads the input channel count from the encoder stem, and loads the weights with
`strict=True` — so the network in the demo is exactly the network that was trained.

---

## 4. Results recorded during training

From `improved_r34_25d/results.json` (threshold 0.325):

| Metric | Value |
|---|---|
| Test Dice — tumor slices | 0.674 |
| Test Dice — all slices | 0.535 |
| Test IoU — tumor slices | 0.559 |
| Out-of-fold validation Dice | 0.481 |
| 3D Dice (per patient) | 0.482 |
| Median volume error | 119.9% |

Training curves are healthy: loss falls 1.01 → 0.27 over 80 epochs and validation Dice rises
steadily to ~0.75 at slice level. The weak numbers are in the **3D** metrics — predicted volumes are
roughly twice the ground truth on every patient listed in `volumetrics.csv`.

---

## 5. Reproducibility check (important)

Those checkpoints were re-evaluated against the dataset masks in the app. The best result obtained
was **≈0.08 Dice** (0.12 at a 0.10 threshold), against the 0.674 recorded above.

The search covered:

- 9-channel encodings: three slices repeated, tiled, interleaved, current-slice-only
- intensity scaling: 0–255, 0–1, ImageNet mean/std, per-slice z-score, percentile-clipped variants
- input sizes: 256 and 384
- thresholds: 0.10 to 0.80
- all eight flip / rotate / transpose orientations

Nothing exceeded ~0.12. The weights load strictly (so the architecture matches) and the model is
extremely confident (peak probability 100%, mean confidence 70–94%) while landing in the wrong
place. **Conclusion: the saved weights do not reproduce the reported scores on this data.** The
original training script was not part of the saved artifacts, so the exact training-time input
assembly cannot be recovered.

This is reported rather than hidden: the app's *Notes & caveats* tab states it, and the demo shows
Dice against ground truth on every slice so the gap is visible.

---

## 6. What was built

| File | Purpose |
|---|---|
| `app.py` | Streamlit demo: dataset browser, upload inference, training results, notes |
| `train.py` | Retraining pipeline: grouped 5-fold CV, out-of-fold threshold selection, Dice3D + volume error |
| `train_colab.ipynb` | Colab notebook that runs `train.py` on a free T4 GPU |
| `README.md` | Setup, run instructions, limitations |
| `<run>-*/` | Original configs, weights, histories, metrics and figures |

**Evaluation protocol in `train.py`** — the part that makes new numbers defensible:

- grouped K-fold over patients
- threshold swept on each fold's **validation** split only, then applied once
- per-patient Dice3D and volume error accumulated from the same validation pass, so every patient is
  scored exactly once (out of fold)
- `IMG_SIZE`, spacing, seed and git commit written into every run's `config.json`

---

## 7. Demo script (2 minutes)

1. Open the app: `python -m streamlit run app.py --server.address 127.0.0.1`
2. Read the blue box: what the system does.
3. Point at the **model card**: UNet, ResNet-34 encoder, scSE attention, 24.6 M parameters, 9-channel
   2.5D input, exact checkpoint filename.
4. In **Dataset**, pick a patient and slice. Walk through the four panels: MRI → ground truth →
   probability map → overlap (green truth, red prediction, yellow agreement).
5. Read the five numbers with the caption underneath: Dice, IoU, predicted vs ground-truth area,
   mean confidence. Explain that high confidence with low Dice means confidently wrong.
6. Click **Evaluate all tumor slices of this patient** to show mean slice Dice and volume error.
7. Open **Training results** for the original metrics, curves and ablation table.
8. Open **Notes & caveats** and state the reproducibility finding honestly.

---

## 8. Limitations and next steps

**Limitations**

- The shipped weights do not reproduce the reported Dice (section 5) — the demo is illustrative.
- No voxel spacing in the configs, so volumes are pixel counts, not mm³.
- 2.5D input depends on slice order; a single uploaded image fills all three positions.
- Trained on one dataset/sequence; other MRI contrasts are out of distribution.
- Research demo only — not a diagnostic tool.

**Next steps**

1. Retrain with `train.py` on Colab (5 folds × 25 epochs, ~2.5–4 h on a T4).
2. Keep threshold selection on out-of-fold validation so the reported number survives review.
3. Attack over-segmentation: largest-component / hysteresis post-processing tuned against Dice3D.
4. Record voxel spacing so volumes can be reported in mm³.
5. Finish the remaining folds for both EfficientNet runs (2 of 5 saved).
