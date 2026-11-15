"""Train 2.5D UNet models for brain tumor segmentation on Kaggle TCGA-LGG.

Protocol
--------
Grouped K-fold cross-validation over *patients* (slices of one patient never
straddle two splits). For each fold the probability threshold is chosen on the
validation split only — sweeping it to maximise slice Dice on tumor slices — and
then applied once. Per-patient Dice3D and volume error come from the same
validation pass, so every patient is scored exactly once (out of fold). No score
in the output is ever tuned on data the model was not validated on.

Example
-------
    python train.py --data archive/kaggle_3m --out runs/r34_25d --epochs 25
    python train.py --data archive/kaggle_3m --out runs/smoke --epochs 1 \
        --folds 1 --limit-patients 8 --size 256 --batch 2 --max-batches 2 \
        --encoder-weights none
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

cv2.setNumThreads(0)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.95, 0.025), 3)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def find_dataset_root(path: str) -> str:
    """Accept the dataset root, its parent, or a folder holding it."""
    if glob.glob(os.path.join(path, "TCGA_*")):
        return path
    for candidate in sorted(glob.glob(os.path.join(path, "*"))):
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "TCGA_*")):
            return candidate
    raise SystemExit(f"No TCGA_* patient folders found under {path!r}")


def patient_files(patient_dir: str) -> list[str]:
    files = [
        p
        for p in glob.glob(os.path.join(patient_dir, "**", "*.tif"), recursive=True)
        if not p.endswith("_mask.tif")
    ]
    files.sort(key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
    return files


def load_patient(patient_dir: str, size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Slices as uint8 clipped to the 1–99 percentile, masks as 0/1 uint8."""
    cache = load_patient.__dict__.setdefault("_cache", {})
    key = (patient_dir, size)
    if key in cache:
        return cache[key]

    images, masks = [], []
    for path in patient_files(patient_dir):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3:
            image = image.astype(np.float32).mean(axis=2)
        image = image.astype(np.float32)
        if image.max() > 255:
            image /= 257.0
        mask = cv2.imread(path[:-4] + "_mask.tif", cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) if mask is not None else np.zeros(image.shape, np.uint8)

        low, high = np.percentile(image, (1, 99))
        image = np.clip(image, low, high)
        images.append(
            cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)
        )
        masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))

    cache.clear() if len(cache) > 16 else None
    cache[key] = (images, masks)
    return images, masks


def augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    turns = random.choice([0, 1, 2, 3])
    if turns:
        image, mask = np.rot90(image, turns, (1, 2)), np.rot90(mask, turns, (1, 2))
    if random.random() < 0.3:  # intensity jitter, keeps the 0–255 contract
        image = image * random.uniform(0.9, 1.1) + random.uniform(-10, 10)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class SliceDataset(Dataset):
    """2.5D samples: three neighbouring slices tiled to nine channels."""

    def __init__(self, items: list[tuple[str, int]], size: int, training: bool):
        self.items = items
        self.size = size
        self.training = training

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        patient_dir, slice_index = self.items[index]
        images, masks = load_patient(patient_dir, self.size)
        count = len(images)
        stack = np.stack(
            [
                images[max(slice_index - 1, 0)],
                images[slice_index],
                images[min(slice_index + 1, count - 1)],
            ],
            axis=0,
        ).astype(np.float32)
        mask = masks[slice_index][None].astype(np.float32)

        if self.training:
            stack, mask = augment(stack, mask)

        # nine channels: the three slices tiled three times
        image = np.tile(np.array(stack, copy=True, order="C"), (3, 1, 1))
        target = np.array(mask, copy=True, order="C")
        return torch.from_numpy(image), torch.from_numpy(target)


def build_items(patient_dirs: list[str], size: int) -> tuple[list[tuple[str, int]], list[bool]]:
    items, tumor = [], []
    for patient_dir in patient_dirs:
        _, masks = load_patient(patient_dir, size)
        for index, mask in enumerate(masks):
            items.append((patient_dir, index))
            tumor.append(bool(mask.sum() > 0))
    return items, tumor


# --------------------------------------------------------------------------- #
# Losses / metrics
# --------------------------------------------------------------------------- #
class HybridLoss(torch.nn.Module):
    def __init__(self, alpha: float, beta: float, focal_alpha: float, gamma: float,
                 weights: tuple[float, float]):
        super().__init__()
        self.tversky = smp.losses.TverskyLoss(mode="binary", alpha=alpha, beta=beta)
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=focal_alpha, gamma=gamma)
        self.weights = weights

    def forward(self, logits, target):
        return self.weights[0] * self.tversky(logits, target) + self.weights[1] * self.focal(
            logits, target
        )


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    total = pred_b.sum() + gt_b.sum()
    return 1.0 if total == 0 else 2.0 * (pred_b & gt_b).sum() / total


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    union = (pred_b | gt_b).sum()
    return 1.0 if union == 0 else (pred_b & gt_b).sum() / union


def postprocess(prob: np.ndarray, threshold: float, min_size: int,
                keep_largest: bool) -> np.ndarray:
    mask = (prob >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    cleaned = np.zeros_like(mask)
    if keep_largest:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[best, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == best] = 1
        return cleaned
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label] = 1
    return cleaned


def autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=True)


@torch.no_grad()
def predict(model, image: torch.Tensor, device: torch.device, tta: bool) -> np.ndarray:
    tensor = image.unsqueeze(0).to(device)
    prob = torch.sigmoid(model(tensor))
    if tta:
        for dims in ((2,), (3,), (2, 3)):
            flipped = torch.flip(tensor, dims=dims)
            prob = prob + torch.flip(torch.sigmoid(model(flipped)), dims=dims)
        prob = prob / 4.0
    return prob[0, 0].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, optimizer, scaler, criterion, device, amp,
                max_batches: int | None) -> float:
    model.train()
    running, seen = 0.0, 0
    for step, (images, masks) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            loss = criterion(model(images), masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


@torch.no_grad()
def val_metrics(model, items, size, device, tta: bool = False,
                max_batches: int | None = None) -> dict:
    """Slice Dice/IoU at a fixed 0.5 threshold, for model selection."""
    model.eval()
    dataset = SliceDataset(items, size, training=False)
    tumor_scores, all_scores, iou_scores = [], [], []
    for step, index in enumerate(range(len(dataset))):
        if max_batches and step >= max_batches:
            break
        image, mask = dataset[index]
        prob = predict(model, image, device, tta)
        gt = mask[0].numpy() > 0.5
        pred = prob >= 0.5
        score = dice(pred, gt)
        all_scores.append(score)
        if gt.any():
            tumor_scores.append(score)
        iou_scores.append(iou(pred, gt))
    return {
        "dice_tumor": float(np.mean(tumor_scores)) if tumor_scores else 0.0,
        "dice_all": float(np.mean(all_scores)) if all_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
    }


@torch.no_grad()
def evaluate_fold(model, items, size, device, min_size: int, keep_largest: bool,
                  tta: bool) -> dict:
    """One pass over the validation fold.

    Probabilities are kept as float16 so the threshold sweep and the final
    post-processed metrics share a single inference pass.
    """
    model.eval()
    dataset = SliceDataset(items, size, training=False)
    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []

    for index in range(len(dataset)):
        image, mask = dataset[index]
        probs.append(predict(model, image, device, tta).astype(np.float16))
        gts.append(mask[0].numpy() > 0.5)

    # 1) sweep the threshold on this fold's validation slices
    best_threshold, best_score = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        scores = [dice(p >= threshold, g) for p, g in zip(probs, gts) if g.any()]
        if scores and np.mean(scores) > best_score:
            best_threshold, best_score = float(threshold), float(np.mean(scores))

    # 2) final metrics at the chosen threshold, with post-processing
    per_patient: dict[str, dict[str, int]] = {}
    tumor_dice, all_dice, tumor_iou = [], [], []
    for (patient_dir, _), prob, gt in zip(items, probs, gts):
        pred = postprocess(prob.astype(np.float32), best_threshold, min_size, keep_largest) > 0
        stats = per_patient.setdefault(
            os.path.basename(patient_dir), {"pred": 0, "gt": 0, "tp": 0, "slices": 0}
        )
        stats["pred"] += int(pred.sum())
        stats["gt"] += int(gt.sum())
        stats["tp"] += int((pred & gt).sum())
        stats["slices"] += 1

        score = dice(pred, gt)
        all_dice.append(score)
        if gt.any():
            tumor_dice.append(score)
            tumor_iou.append(iou(pred, gt))

    volumetrics, dice3d = [], []
    for patient, stats in sorted(per_patient.items()):
        if stats["gt"] == 0:
            continue
        score = 2.0 * stats["tp"] / max(stats["pred"] + stats["gt"], 1)
        dice3d.append(score)
        volumetrics.append(
            {
                "patient": patient,
                "n_slices": stats["slices"],
                "dice3d": round(score, 4),
                "pred_vol": stats["pred"],
                "gt_vol": stats["gt"],
                "abs_vol_err_%": round(abs(stats["pred"] - stats["gt"]) / stats["gt"] * 100, 2),
            }
        )

    return {
        "threshold": best_threshold,
        "dice_tumor": float(np.mean(tumor_dice)) if tumor_dice else 0.0,
        "dice_all": float(np.mean(all_dice)) if all_dice else 0.0,
        "iou_tumor": float(np.mean(tumor_iou)) if tumor_iou else 0.0,
        "dice3d": float(np.mean(dice3d)) if dice3d else 0.0,
        "vol_err_median": float(np.median([v["abs_vol_err_%"] for v in volumetrics]))
        if volumetrics
        else 0.0,
        "volumetrics": volumetrics,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="archive/kaggle_3m", help="dataset root or a folder holding it")
    parser.add_argument("--out", default="runs/experiment", help="output run folder")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet", help="imagenet or none")
    parser.add_argument("--attention", default="scse", help="scse, cbam or none")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5, help="number of CV folds")
    parser.add_argument("--run-folds", default="all", help="e.g. 0,1 or all")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr-decoder", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--oversample", type=float, default=2.5, help="weight of tumor slices")
    parser.add_argument("--tversky", default="0.3,0.7")
    parser.add_argument("--focal", default="0.75,2.0")
    parser.add_argument("--min-size", type=int, default=40)
    parser.add_argument("--no-keep-largest", action="store_true")
    parser.add_argument("--val-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-patients", type=int, default=0, help="debug: use this many patients")
    parser.add_argument("--max-batches", type=int, default=0, help="debug: cap batches per epoch")
    parser.add_argument("--patience", type=int, default=0, help="early stop after N epochs without gain")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp = (device.type == "cuda") and not args.no_amp
    dataset_root = find_dataset_root(args.data)
    patient_dirs = sorted(
        p for p in glob.glob(os.path.join(dataset_root, "TCGA_*")) if os.path.isdir(p)
    )
    if args.limit_patients:
        patient_dirs = patient_dirs[: args.limit_patients]
    if not patient_dirs:
        raise SystemExit("No patients found.")

    ids = [os.path.basename(p) for p in patient_dirs]
    random.Random(args.seed).shuffle(ids)
    by_id = dict(zip(ids, patient_dirs))
    fold_ids = [list(chunk) for chunk in np.array_split(np.array(ids), args.folds)]

    os.makedirs(args.out, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.out))
    print(f"run={run_name}  device={device}  amp={amp}")
    print(f"patients={len(ids)}  folds={args.folds}  size={args.size}  epochs={args.epochs}")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = None

    config = {
        "RUN_NAME": run_name,
        "SEED": args.seed,
        "KFOLD": args.folds,
        "USE_2_5D": True,
        "IMG_SIZE": args.size,
        "SPACING_MM": None,
        "PCT_CLIP": [1, 99],
        "POS_OVERSAMPLE": args.oversample,
        "AUG": True,
        "BATCH": args.batch,
        "ARCH": "smp_unet",
        "ENCODER": args.encoder,
        "PRETRAINED": args.encoder_weights,
        "ATTENTION": args.attention,
        "REGION_LOSS": "tversky+focal",
        "TVERSKY_AB": [float(v) for v in args.tversky.split(",")],
        "FOCAL_AG": [float(v) for v in args.focal.split(",")],
        "EPOCHS": args.epochs,
        "LR_DECODER": args.lr_decoder,
        "LR_ENCODER": args.lr_encoder,
        "WEIGHT_DECAY": args.weight_decay,
        "WARMUP_EPOCHS": args.warmup,
        "AMP": amp,
        "TTA": 4 if args.val_tta else 1,
        "THRESHOLD_SWEEP": [float(THRESHOLD_GRID[0]), float(THRESHOLD_GRID[-1])],
        "POST_MIN_SIZE": args.min_size,
        "KEEP_LARGEST": not args.no_keep_largest,
        "DATASET": dataset_root,
        "PATIENTS": len(ids),
        "GIT_COMMIT": commit,
        "CREATED": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    selected = (
        list(range(args.folds))
        if args.run_folds == "all"
        else [int(v) for v in args.run_folds.split(",")]
    )
    per_fold, all_volumetrics = [], []

    for fold in selected:
        val_patients = [by_id[i] for i in fold_ids[fold]]
        train_patients = [by_id[i] for i in ids if i not in fold_ids[fold]]
        if not train_patients or not val_patients:
            continue

        train_items, train_tumor = build_items(train_patients, args.size)
        val_items, _ = build_items(val_patients, args.size)
        weights = [args.oversample if tumor else 1.0 for tumor in train_tumor]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            SliceDataset(train_items, args.size, training=True),
            batch_size=args.batch, sampler=sampler, num_workers=args.workers,
            pin_memory=(device.type == "cuda"), drop_last=len(train_items) > args.batch,
        )

        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
            in_channels=9, classes=1,
            decoder_attention_type=None if args.attention == "none" else args.attention,
        ).to(device)
        encoder_params = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": args.lr_encoder},
                {"params": [p for p in model.parameters() if id(p) not in encoder_ids],
                 "lr": args.lr_decoder},
            ],
            weight_decay=args.weight_decay,
        )
        total_epochs = max(args.epochs, 1)

        def schedule(epoch: int) -> float:
            if args.warmup and epoch < args.warmup:
                return (epoch + 1) / args.warmup
            progress = (epoch - args.warmup) / max(total_epochs - args.warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        criterion = HybridLoss(
            *[float(v) for v in args.tversky.split(",")],
            *[float(v) for v in args.focal.split(",")],
            (1.0, 1.0),
        )
        scaler = torch.amp.GradScaler("cuda") if amp else None

        history_path = os.path.join(args.out, f"history_fold{fold}.csv")
        with open(history_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_loss", "lr", "dice_all", "dice_tumor", "iou_all", "iou_tumor"])

            best_score, best_epoch, stale = -1.0, 0, 0
            for epoch in range(1, total_epochs + 1):
                started = time.time()
                loss = train_epoch(
                    model, train_loader, optimizer, scaler, criterion, device, amp,
                    args.max_batches or None,
                )
                metrics = val_metrics(
                    model, val_items, args.size, device, tta=False,
                    max_batches=args.max_batches or None,
                )
                lr = optimizer.param_groups[-1]["lr"]
                writer.writerow([epoch, round(loss, 6), f"{lr:.3e}", round(metrics["dice_all"], 6),
                                 round(metrics["dice_tumor"], 6), round(metrics["iou"], 6),
                                 round(metrics["iou"], 6)])
                fh.flush()

                print(
                    f"[fold {fold}] epoch {epoch}/{total_epochs} loss={loss:.4f} "
                    f"dice_tumor={metrics['dice_tumor']:.4f} dice_all={metrics['dice_all']:.4f} "
                    f"({time.time() - started:.1f}s)"
                )
                scheduler.step()

                if metrics["dice_tumor"] > best_score:
                    best_score, best_epoch, stale = metrics["dice_tumor"], epoch, 0
                    torch.save(model.state_dict(), os.path.join(args.out, f"best_fold{fold}.pt"))
                else:
                    stale += 1
                    if args.patience and stale >= args.patience:
                        print(f"[fold {fold}] early stop at epoch {epoch}")
                        break

        weights_path = os.path.join(args.out, f"best_fold{fold}.pt")
        if not os.path.exists(weights_path):
            torch.save(model.state_dict(), weights_path)
            best_epoch = epoch
        print(f"[fold {fold}] best epoch {best_epoch} (selection dice {best_score:.4f}) — evaluating")

        model.load_state_dict(torch.load(weights_path, map_location=device))
        result = evaluate_fold(
            model, val_items, args.size, device, args.min_size,
            keep_largest=not args.no_keep_largest, tta=args.val_tta,
        )
        for row in result["volumetrics"]:
            all_volumetrics.append({"fold": fold, **row})
        per_fold.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "selection_dice_tumor": round(float(best_score), 4),
                "threshold": result["threshold"],
                "dice_tumor": round(result["dice_tumor"], 4),
                "dice_all": round(result["dice_all"], 4),
                "iou_tumor": round(result["iou_tumor"], 4),
                "dice3d": round(result["dice3d"], 4),
                "vol_err_median": round(result["vol_err_median"], 2),
            }
        )
        print(f"[fold {fold}] " + json.dumps(per_fold[-1]))

    if not per_fold:
        raise SystemExit("No fold was trained.")

    mean = lambda key: float(np.mean([row[key] for row in per_fold]))  # noqa: E731
    results = {
        "run": run_name,
        "folds": [row["fold"] for row in per_fold],
        "img_size": args.size,
        "spacing_mm": None,
        "threshold": round(mean("threshold"), 4),
        "oof": {
            "dice_tumor": round(mean("dice_tumor"), 4),
            "dice_all": round(mean("dice_all"), 4),
            "iou_tumor": round(mean("iou_tumor"), 4),
            "dice3d": round(mean("dice3d"), 4),
            "vol_err_median": round(mean("vol_err_median"), 2),
        },
        "per_fold": per_fold,
        "note": "Threshold chosen per fold on its own validation split; every patient scored once.",
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(args.out, "threshold.json"), "w", encoding="utf-8") as fh:
        json.dump({"threshold": results["threshold"]}, fh)
    if all_volumetrics:
        with open(os.path.join(args.out, "volumetrics.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_volumetrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_volumetrics)

    print("\nOOF results:", json.dumps(results["oof"], indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
