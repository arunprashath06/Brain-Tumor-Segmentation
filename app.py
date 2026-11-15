"""Brain tumor segmentation — Streamlit front end.

2.5D UNet (segmentation-models-pytorch) checkpoints trained on the Kaggle
TCGA-LGG brain MRI dataset.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import segmentation_models_pytorch as smp

ROOT = os.path.dirname(os.path.abspath(__file__))

# Values taken from config.json / the saved figures in the run folders. The
# checkpoints carry no metadata, so these are the documented defaults.
REF_SIZE = 384
REF_MIN_SIZE = 40
DATASET_DIR = os.path.join(ROOT, "archive", "kaggle_3m")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_runs() -> list[dict]:
    """Every best_fold*.pt that sits next to a config.json."""
    runs: list[dict] = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "*", "*", "config.json"))):
        run_dir = os.path.dirname(cfg_path)
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        weights = sorted(glob.glob(os.path.join(run_dir, "best_fold*.pt")))
        if not weights:
            continue

        threshold = 0.5
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, encoding="utf-8") as fh:
                    threshold = float(json.load(fh).get("threshold", 0.5))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        run_name = cfg.get("RUN_NAME", os.path.basename(run_dir))
        for weight_path in weights:
            fold = os.path.basename(weight_path)[len("best_fold"):-len(".pt")]
            runs.append(
                {
                    "label": f"{run_name} · fold {fold}",
                    "run": run_name,
                    "fold": fold,
                    "dir": run_dir,
                    "weights": os.path.relpath(weight_path, ROOT),
                    "encoder": cfg.get("ENCODER", "resnet34"),
                    "attention": cfg.get("ATTENTION"),
                    "threshold": threshold,
                    "config": cfg,
                }
            )
    return runs


@st.cache_data(show_spinner=False)
def list_patients() -> list[str]:
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(
        path for path in glob.glob(os.path.join(DATASET_DIR, "TCGA_*")) if os.path.isdir(path)
    )


@st.cache_data(show_spinner="Reading patient slices…")
def load_patient(patient_dir: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Return (images, masks, names) for one patient, ordered by slice number."""
    files = [
        path
        for path in glob.glob(os.path.join(patient_dir, "**", "*.tif"), recursive=True)
        if not path.endswith("_mask.tif")
    ]
    files.sort(key=lambda path: int(os.path.basename(path).rsplit("_", 1)[1].split(".")[0]))

    images, masks, names = [], [], []
    for path in files:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3:
            image = image.astype(np.float32).mean(axis=2)
        mask_path = path[:-4] + "_mask.tif"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        images.append(image.astype(np.float32))
        masks.append((mask > 127).astype(np.uint8) if mask is not None else np.zeros_like(image, dtype=np.uint8))
        names.append(os.path.basename(path))
    return images, masks, names


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
STEM_KEYS = ("encoder.conv1.weight", "encoder._conv_stem.weight")


def infer_in_channels(state_dict: dict) -> int:
    for key in STEM_KEYS:
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise KeyError("Could not locate the encoder stem in this checkpoint.")


def build_model(weights_path: str, encoder: str, attention: str | None):
    """Rebuild the smp UNet described by a checkpoint and load it strictly."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]

    in_channels = infer_in_channels(state)
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        decoder_attention_type=attention or None,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, in_channels


@st.cache_resource(show_spinner="Loading checkpoint…")
def load_model(weights_path: str, encoder: str, attention: str | None, device: str):
    model, in_channels = build_model(os.path.join(ROOT, weights_path), encoder, attention)
    return model.to(device), in_channels


# --------------------------------------------------------------------------- #
# Pre-processing / inference
# --------------------------------------------------------------------------- #
def decode_image(raw: bytes) -> np.ndarray:
    """Decode PNG/JPG/TIF bytes to a float32 grayscale array on a 0–255 scale."""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("That file is not an image Streamlit/OpenCV can decode.")
    if image.ndim == 3:
        image = image[:, :, :3].astype(np.float32).mean(axis=2)
    image = image.astype(np.float32)
    if image.max() > 255:  # 12/16-bit MRI exports
        image /= 257.0
    return image


def prep_slice(gray: np.ndarray, size: int, pct: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Resize and clip to the 1st–99th percentile (config PCT_CLIP: [1, 99]).

    Keeps the 0–255 range: the checkpoints respond to intensity magnitude, so
    dividing by 255 silently flattened every prediction to a constant.
    """
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(resized, pct)
    return np.clip(resized, low, high).astype(np.float32)


def build_input(slices: list[np.ndarray], index: int, in_channels: int) -> np.ndarray:
    """Three neighbouring slices, expanded to the channel count the stem expects."""
    previous = slices[max(index - 1, 0)]
    current = slices[index]
    following = slices[min(index + 1, len(slices) - 1)]
    stack = np.stack([previous, current, following], axis=0).astype(np.float32)

    if in_channels == 3:
        return stack
    if in_channels % 3 == 0:
        return np.tile(stack, (in_channels // 3, 1, 1))
    raise ValueError(f"Unsupported input channel count: {in_channels}")


def predict_prob(model, volume: np.ndarray, device: str, tta: bool) -> np.ndarray:
    tensor = torch.from_numpy(volume).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))
        if tta:  # 4-way flip TTA, as in config TTA=4
            for dims in ((2,), (3,), (2, 3)):
                flipped = torch.flip(tensor, dims=dims)
                prob = prob + torch.flip(torch.sigmoid(model(flipped)), dims=dims)
            prob = prob / 4.0
    return prob[0, 0].detach().cpu().numpy()


def postprocess(prob: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    mask = (prob >= threshold).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, labels_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label] = 1
    return cleaned


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    total = pred_b.sum() + gt_b.sum()
    return 1.0 if total == 0 else 2.0 * (pred_b & gt_b).sum() / total


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    union = (pred_b | gt_b).sum()
    return 1.0 if union == 0 else (pred_b & gt_b).sum() / union


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def to_rgb(gray: np.ndarray) -> np.ndarray:
    scaled = gray - gray.min()
    scaled = scaled / max(scaled.max(), 1e-6)
    return np.repeat(scaled[..., None], 3, axis=2)


def overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = to_rgb(gray)
    hit = mask.astype(bool)
    if hit.any():
        rgb = np.where(hit[..., None], rgb * 0.55 + np.array([1.0, 0.15, 0.15]) * 0.45, rgb)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (1.0, 0.0, 0.0), 2)
    return np.clip(rgb, 0.0, 1.0)


def compare_overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Green = ground truth, red = prediction, yellow = overlap."""
    rgb = to_rgb(gray).copy()
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    rgb[gt_b] = rgb[gt_b] * 0.45 + np.array([0.2, 1.0, 0.3]) * 0.55
    rgb[pred_b] = rgb[pred_b] * 0.45 + np.array([1.0, 0.2, 0.2]) * 0.55
    return np.clip(rgb, 0.0, 1.0)


def heatmap(prob: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0


def model_card(run: dict, in_channels: int, params: int) -> None:
    left, mid, right = st.columns([2, 2, 1])
    left.markdown(
        f"**Model** · UNet + scSE attention  \n"
        f"**Encoder** · `{run['encoder']}` · **Params** · {params / 1e6:.1f} M"
    )
    mid.markdown(
        f"**Input** · {in_channels} ch · 2.5D (3 slices)  \n"
        f"**Checkpoint** · `{run['weights']}`"
    )
    right.markdown(f"**Fold** · {run['fold']}  \n**Run** · `{run['run']}`")
    st.caption(
        "How it works: a **UNet** — an encoder (`{}`) that compresses the slice into features, and a "
        "decoder that rebuilds a tumour mask from them. **scSE attention** re-weights the most useful "
        "channels and regions. It sees **{}-channel 2.5D input** (three neighbouring slices tiled). "
        "This exact checkpoint holds **{:.1f} M** learned parameters.".format(
            run['encoder'], in_channels, params / 1e6
        )
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide")

    runs = discover_runs()
    if not runs:
        st.warning(
            "No model checkpoint found in this deployment.\n\n"
            "Expected `<run>/config.json` next to `best_fold*.pt`. The weights are ~90 MB each and "
            "are kept out of git; see DEPLOY.md for adding one with Git LFS."
        )
        return

    patients = list_patients()

    st.title("🧠 Brain Tumor Segmentation and Volumetric Delineation")
    st.caption("2.5D UNet · scSE attention · Kaggle TCGA-LGG brain MRI · Flair slices, axial view.")
    st.info(
        "**What this is:** a deep-learning model that looks at one brain MRI slice and outlines any "
        "tumour it finds. Pick a patient below (or upload your own image) and the app shows the "
        "MRI, the model's probability map, the final outline, and — for dataset slices — how that "
        "outline compares with the radiologist's ground truth."
    )

    labels = [run["label"] for run in runs]
    default_index = next(
        (i for i, run in enumerate(runs) if os.path.exists(os.path.join(run["dir"], "results.json"))), 0
    )

    with st.sidebar:
        st.header("Model")
        choice = st.selectbox("Checkpoint", labels, index=default_index)
        run = runs[labels.index(choice)]

        device = st.selectbox("Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0)
        size = st.select_slider("Input size", options=[256, 320, 384, 448, 512], value=REF_SIZE)
        threshold = st.slider(
            "Probability threshold", 0.05, 0.95,
            value=float(min(max(run["threshold"], 0.05), 0.95)), step=0.025,
            help="Lower values predict more tumor. The value saved with the run is the default.",
        )
        use_tta = st.checkbox("Flip TTA (4×, slower)", value=True)
        min_size_base = st.slider(
            "Drop components smaller than (px)", 0, 400, REF_MIN_SIZE, step=10,
            help="POST_MIN_SIZE from the training config, scaled to the input size.",
        )
        with st.expander("Run config"):
            st.json(run["config"])

    min_size = max(1, round(min_size_base * (size / REF_SIZE) ** 2))
    model, in_channels = load_model(run["weights"], run["encoder"], run["attention"], device)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    model_card(run, in_channels, n_params)
    st.divider()

    tab_data, tab_upload, tab_perf, tab_notes = st.tabs(
        ["Dataset", "Upload", "Training results", "Notes & caveats"]
    )

    # ---------------------------------------------------------------- dataset
    with tab_data:
        if not patients:
            st.warning(
                "No dataset found. Expected the Kaggle LGG folder at `archive/kaggle_3m/TCGA_*`."
            )
        else:
            pick, side = st.columns([3, 2])
            patient_dir = pick.selectbox(
                "Patient", patients, index=0, format_func=lambda path: os.path.basename(path)
            )
            images, masks, names = load_patient(patient_dir)
            tumor_slices = [i for i, mask in enumerate(masks) if mask.sum() > 0]

            default_slice = tumor_slices[len(tumor_slices) // 2] if tumor_slices else len(images) // 2
            index = side.slider("Slice", 1, len(images), default_slice + 1) - 1

            slices = [prep_slice(image, size) for image in images]
            prob = predict_prob(model, build_input(slices, index, in_channels), device, use_tta)
            mask = postprocess(prob, threshold, min_size)

            gt = (
                cv2.resize(masks[index], (size, size), interpolation=cv2.INTER_NEAREST) > 0
            ).astype(np.uint8)

            coverage = mask.mean() * 100
            gt_coverage = gt.mean() * 100
            mean_conf = float(prob[mask.astype(bool)].mean()) if mask.any() else 0.0

            st.markdown(
                f"**{names[index]}** — "
                + (
                    f"tumor predicted on **{coverage:.2f}%** of the slice "
                    f"(mean confidence **{mean_conf * 100:.0f}%**, peak **{prob.max() * 100:.0f}%**)"
                    if mask.any()
                    else "**no tumor predicted** at this threshold"
                )
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Dice vs GT", f"{dice(mask, gt):.3f}", help="Overlap with the ground truth: 1.0 = identical, 0.0 = no overlap at all.")
            m2.metric("IoU vs GT", f"{iou(mask, gt):.3f}", help="Same idea as Dice, stricter: intersection ÷ union.")
            m3.metric("Predicted area", f"{coverage:.2f}%", help="How much of this slice the model marked as tumour.")
            m4.metric("Ground-truth area", f"{gt_coverage:.2f}%", help="How much of this slice the radiologist marked as tumour.")
            m5.metric("Mean confidence", f"{mean_conf * 100:.0f}%", help="Average probability inside the predicted outline.")
            st.caption(
                "**How to read these numbers** — *Dice* and *IoU* compare the model's outline with the "
                "radiologist's; above ~0.7 is good, and 0 means the two do not overlap anywhere. "
                "*Predicted area* vs *ground-truth area* shows whether the model over- or under-segments: "
                "close numbers mean similar size. *Mean confidence* is how sure the model is inside its "
                "own outline — high confidence with low Dice means it is confidently wrong."
            )

            v1, v2, v3, v4 = st.columns(4)
            v1.image(to_rgb(images[index]), caption="1 · Input — the MRI slice", width="stretch")
            v2.image((gt * 255).astype(np.uint8), caption="2 · Ground truth — radiologist's tumour outline", clamp=True, width="stretch")
            v3.image(heatmap(prob), caption="3 · Model probability — brighter means 'more likely tumour'", width="stretch")
            v4.image(compare_overlay(slices[index], gt, mask), caption="4 · Overlap — green = ground truth, red = model, yellow = both agree", width="stretch")

            if tumor_slices:
                if st.button(f"Evaluate all {len(tumor_slices)} tumor slices of this patient"):
                    rows = []
                    bar = st.progress(0.0, text="Running…")
                    gt_total = pred_total = 0
                    for order, slice_index in enumerate(tumor_slices, start=1):
                        prob_i = predict_prob(
                            model, build_input(slices, slice_index, in_channels), device, use_tta
                        )
                        mask_i = postprocess(prob_i, threshold, min_size)
                        gt_i = (
                            cv2.resize(masks[slice_index], (size, size), interpolation=cv2.INTER_NEAREST) > 0
                        ).astype(np.uint8)
                        rows.append(
                            {
                                "slice": names[slice_index],
                                "dice": round(dice(mask_i, gt_i), 3),
                                "pred_area_%": round(float(mask_i.mean() * 100), 2),
                                "gt_area_%": round(float(gt_i.mean() * 100), 2),
                            }
                        )
                        gt_total += int(gt_i.sum())
                        pred_total += int(mask_i.sum())
                        bar.progress(order / len(tumor_slices), text=f"Slice {order}/{len(tumor_slices)}")
                    bar.empty()

                    frame = pd.DataFrame(rows)
                    st.dataframe(frame, hide_index=True)
                    e1, e2 = st.columns(2)
                    e1.metric("Mean slice Dice (tumor slices)", f"{frame['dice'].mean():.3f}")
                    volume_error = abs(pred_total - gt_total) / max(gt_total, 1) * 100
                    e2.metric("Volume error (pixels)", f"{volume_error:.0f}%")
                    st.caption(
                        "Volume error compares predicted and ground-truth tumor pixels summed over these "
                        "slices. Voxel→mm³ needs spacing, which the training config leaves as null."
                    )

    # ----------------------------------------------------------------- upload
    with tab_upload:
        uploads = st.file_uploader(
            "Upload MRI slices (PNG / JPG / BMP / TIF). Several slices are treated as one volume.",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if not uploads:
            st.info(
                "Upload one axial slice for a quick check, or several slices of the same scan. "
                "Slice order is by filename, so prefix names (`slice_001.png`) for a stack."
            )
        else:
            files = sorted(uploads, key=lambda file: file.name)
            try:
                slices = [prep_slice(decode_image(file.getvalue()), size) for file in files]
            except ValueError as exc:
                st.error(str(exc))
                return

            index = 0
            if len(files) > 1:
                index = st.slider("Slice to display", 1, len(files), len(files) // 2) - 1

            prob = predict_prob(model, build_input(slices, index, in_channels), device, use_tta)
            mask = postprocess(prob, threshold, min_size)
            mean_conf = float(prob[mask.astype(bool)].mean()) if mask.any() else 0.0

            st.markdown(
                f"**Output** — "
                + (
                    f"tumor on **{mask.mean() * 100:.2f}%** of the slice, mean confidence "
                    f"**{mean_conf * 100:.0f}%**, peak **{prob.max() * 100:.0f}%**"
                    if mask.any()
                    else f"**no tumor predicted** at threshold {threshold:.3f} "
                         f"(peak probability {prob.max() * 100:.0f}%)"
                )
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.image(to_rgb(slices[index]), caption="1 · Input — your MRI slice", width="stretch")
            c2.image(heatmap(prob), caption="2 · Model probability — brighter means 'more likely tumour'", width="stretch")
            c3.image((mask * 255).astype(np.uint8), caption=f"3 · Final mask — white where probability ≥ {threshold:.2f}", clamp=True, width="stretch")
            c4.image(overlay(slices[index], mask), caption="4 · Outline drawn on the scan", width="stretch")

            ok, encoded = cv2.imencode(".png", (mask * 255).astype(np.uint8))
            if ok:
                st.download_button(
                    "Download mask PNG", encoded.tobytes(),
                    file_name=f"mask_{run['run']}_f{run['fold']}_{index + 1}.png", mime="image/png",
                )

            if len(files) > 1 and st.button("Segment every uploaded slice"):
                areas = []
                bar = st.progress(0.0, text="Segmenting…")
                for position in range(len(slices)):
                    slice_prob = predict_prob(
                        model, build_input(slices, position, in_channels), device, use_tta
                    )
                    areas.append(int(postprocess(slice_prob, threshold, min_size).sum()))
                    bar.progress((position + 1) / len(slices), text=f"Slice {position + 1}/{len(slices)}")
                bar.empty()
                st.line_chart(pd.DataFrame({"slice": range(1, len(areas) + 1), "tumor_px": areas}).set_index("slice"))
                st.caption(f"{sum(areas)} positive pixels across {sum(a > 0 for a in areas)} slices.")

    # -------------------------------------------------------------- performance
    with tab_perf:
        st.subheader("Held-out test metrics (from the original run)")
        rows = []
        for seen in sorted({r["run"]: r for r in runs}.values(), key=lambda r: r["run"]):
            results_path = os.path.join(seen["dir"], "results.json")
            if not os.path.exists(results_path):
                continue
            with open(results_path, encoding="utf-8") as fh:
                results = json.load(fh)
            row = {"run": seen["run"], "threshold": results.get("threshold")}
            row.update({f"test_{k}": v for k, v in results.get("test", {}).items()})
            row.update({f"val_oof_{k}": v for k, v in results.get("val_oof", {}).items()})
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(
                "These are the scores recorded during training. On this machine the same checkpoints "
                "score ≈0.1 Dice against the dataset masks — see Notes & caveats."
            )

        history_files = sorted(glob.glob(os.path.join(run["dir"], "history_fold*.csv")))
        if history_files:
            st.subheader(f"Training curves — {run['run']}")
            picked = st.selectbox("Fold file", [os.path.basename(p) for p in history_files], index=0)
            history = pd.read_csv(os.path.join(run["dir"], picked))
            st.line_chart(history.set_index("epoch")[["dice_all", "dice_tumor", "iou_all", "iou_tumor"]])
            st.caption(f"{len(history)} epochs · final train loss {history['train_loss'].iloc[-1]:.4f}")

        ablation_path = os.path.join(ROOT, "ablation_log.csv")
        if os.path.exists(ablation_path):
            st.subheader("Threshold / config ablation")
            st.dataframe(pd.read_csv(ablation_path), hide_index=True)

        volumetrics_path = os.path.join(run["dir"], "volumetrics.csv")
        if os.path.exists(volumetrics_path):
            st.subheader("Per-patient 3D results")
            volumetrics = pd.read_csv(volumetrics_path)
            st.dataframe(volumetrics, hide_index=True)
            st.caption(
                f"abs_vol_err_% = |predicted − true| / true. Median: {volumetrics['abs_vol_err_%'].median():.1f}%."
            )

        figures = sorted(glob.glob(os.path.join(run["dir"], "*.png")))
        comparison = os.path.join(ROOT, "model_comparison.png")
        if os.path.exists(comparison):
            figures.append(comparison)
        if figures:
            st.subheader("Saved figures")
            columns = st.columns(2)
            for position, path in enumerate(figures):
                with columns[position % 2]:
                    st.image(path, caption=os.path.basename(path), width="stretch")

    # ------------------------------------------------------------------- notes
    with tab_notes:
        st.subheader("How a prediction is built")
        st.markdown(
            f"""
**Checkpoint** — `{run['weights']}`, rebuilt as
`smp.Unet(encoder_name="{run['encoder']}", in_channels={in_channels}, classes=1,`
`decoder_attention_type="{run['attention']}")` and loaded with `strict=True`.

**2.5D input** — the stem takes **{in_channels} channels**: three neighbouring axial slices,
tiled to fill the channel count. A single uploaded image fills all three positions.

**Pre-processing** — grayscale → resize → 1st–99th percentile clip (`PCT_CLIP: [1, 99]`),
kept on the 0–255 intensity scale. Dividing by 255 flattens the output to a constant, so the
scale is part of the model's contract.

**Post-processing** — sigmoid → threshold → connected components below `POST_MIN_SIZE` dropped.

### What the numbers mean
- **Dice** — overlap between the predicted outline and the ground truth, from 0 (none) to 1 (identical).
- **IoU** — the same comparison, stricter: intersection divided by union.
- **Predicted / ground-truth area** — percentage of the slice covered by each mask. Similar values mean the model found a tumour of the right size pointing at the right place; a large gap means under- or over-segmentation.
- **Mean / peak confidence** — average and maximum sigmoid probability. High confidence with low Dice is the model being confidently wrong, which is exactly what the shipped checkpoints do.
- **Threshold** — probability above which a pixel is called tumour. Lowering it grows the mask, raising it shrinks it.

### Reproducibility check
The seven saved checkpoints were evaluated against the Kaggle LGG masks. Across 9-channel
encodings (repeat / tile / current-only), 0–255 vs 0–1 vs ImageNet scaling, 256 and 384 input
sizes, thresholds 0.1–0.8 and all eight image orientations, the best slice Dice reached **≈0.10**,
against **0.674** recorded in `results.json`. The architecture loads strictly, so the weights are
intact — they simply do not reproduce the reported scores on this data. Treat the masks as
illustrative until the models are retrained.

### Other limits
- Slice order defines the 2.5D window; shuffled uploads change the prediction.
- No voxel spacing in the config (`SPACING_MM: null`), so volume is reported in pixels.
- Trained on the LGG T1c/Flair-style slices of one dataset; other sequences are out of distribution.
- Research demo only — not a diagnostic tool.
"""
        )


if __name__ == "__main__":
    main()
