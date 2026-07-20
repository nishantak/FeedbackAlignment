from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


SEED = 42
BOOTSTRAP_SEED = 20260720
BOOTSTRAP_RESAMPLES = 10_000
BATCH_SIZE = 64
TARGET_CLASS_INDEX = 1
TARGET_LAYER_NAME = "stage2.1.conv2"
TOP_PROPORTIONS = (0.10, 0.20)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "cnn_learning_results" / "models"
OUTPUT_DIR = ROOT / "cnn_learning_results" / "attribution_config_1"
TEST_IMAGES = ROOT / "pcam_data" / "test" / "test_img.h5"
TEST_LABELS = ROOT / "pcam_data" / "test" / "test_label.h5"
BP_CHECKPOINT = MODEL_DIR / "backprop_config_1.pt"
DFA_CHECKPOINT = MODEL_DIR / "dfa_config_1.pt"
MANUSCRIPT_FIGURE = ROOT.parent.parent / "paper" / "figs" / "pcam_gradcam_agreement.pdf"


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, use_dropout=False):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.drop = nn.Dropout2d(p=0.2) if use_dropout else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return self.drop(out)


class PCamCNN(nn.Module):
    def __init__(self, num_classes=2, blocks=3, bn_last=True, dropout_last=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.stage1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.stage2 = nn.Sequential(
            BasicBlock(32, 64),
            BasicBlock(64, 64, use_bn=True, use_dropout=(blocks == 2 and dropout_last)),
        )
        if blocks == 3:
            self.stage3 = nn.Sequential(
                nn.MaxPool2d(2),
                BasicBlock(64, 128),
                BasicBlock(128, 128, use_bn=bn_last, use_dropout=dropout_last),
            )
            in_features = 128
        else:
            self.stage3 = nn.Identity()
            in_features = 64
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    safe_globals = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_model(checkpoint):
    model = PCamCNN(**checkpoint["config"])
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Strict load failed: {load_result}")
    return model.eval()


def module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Target layer {name!r} not found")
    return modules[name]


def grad_cam(model: nn.Module, images: torch.Tensor):
    captured = []

    def capture_activation(_module, _inputs, output):
        captured.append(output)

    handle = module_by_name(model, TARGET_LAYER_NAME).register_forward_hook(capture_activation)
    try:
        logits = model(images)
        if len(captured) != 1:
            raise RuntimeError(f"Expected one target activation, got {len(captured)}")
        activation = captured[0]
        gradients = torch.autograd.grad(logits[:, TARGET_CLASS_INDEX].sum(), activation)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        raw_native = (weights * activation).sum(dim=1, keepdim=True)
        native = torch.relu(raw_native)
        image_space = F.interpolate(
            native, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return (
            logits.detach(), raw_native[:, 0].detach(), native[:, 0].detach(),
            image_space[:, 0].detach(),
        )
    finally:
        handle.remove()


def top_mask(values: np.ndarray, proportion: float):
    n_locations = values.shape[1]
    k = math.ceil(proportion * n_locations)
    order = np.argsort(-values, axis=1, kind="stable")
    mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(mask, order[:, :k], True, axis=1)
    return mask, k


def iou_rows(first: np.ndarray, second: np.ndarray, proportion: float):
    first_mask, first_k = top_mask(first, proportion)
    second_mask, second_k = top_mask(second, proportion)
    intersection = np.logical_and(first_mask, second_mask).sum(axis=1)
    union = np.logical_or(first_mask, second_mask).sum(axis=1)
    return intersection / union, first_k, second_k


def spearman_rows(first: np.ndarray, second: np.ndarray):
    first_ranks = rankdata(first, axis=1, method="average")
    second_ranks = rankdata(second, axis=1, method="average")
    first_centered = first_ranks - first_ranks.mean(axis=1, keepdims=True)
    second_centered = second_ranks - second_ranks.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", first_centered, second_centered)
    denominator = np.sqrt(
        np.einsum("ij,ij->i", first_centered, first_centered)
        * np.einsum("ij,ij->i", second_centered, second_centered)
    )
    correlations = np.full(first.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = numerator[valid] / denominator[valid]
    return correlations


def map_metrics(bp_maps: np.ndarray, dfa_maps: np.ndarray, prefix: str):
    bp_flat = bp_maps.reshape(bp_maps.shape[0], -1)
    dfa_flat = dfa_maps.reshape(dfa_maps.shape[0], -1)
    bp_finite = np.isfinite(bp_flat).all(axis=1)
    dfa_finite = np.isfinite(dfa_flat).all(axis=1)
    bp_constant = np.ptp(bp_flat, axis=1) <= 1e-12
    dfa_constant = np.ptp(dfa_flat, axis=1) <= 1e-12
    valid = bp_finite & dfa_finite & ~bp_constant & ~dfa_constant

    iou10, bp_k10, dfa_k10 = iou_rows(bp_flat, dfa_flat, TOP_PROPORTIONS[0])
    iou20, bp_k20, dfa_k20 = iou_rows(bp_flat, dfa_flat, TOP_PROPORTIONS[1])
    spearman = spearman_rows(bp_flat, dfa_flat)
    iou10[~valid] = np.nan
    iou20[~valid] = np.nan
    spearman[~valid] = np.nan
    return {
        f"{prefix}_iou_top10": iou10,
        f"{prefix}_iou_top20": iou20,
        f"{prefix}_spearman": spearman,
        f"{prefix}_bp_constant": bp_constant,
        f"{prefix}_dfa_constant": dfa_constant,
        f"{prefix}_bp_finite": bp_finite,
        f"{prefix}_dfa_finite": dfa_finite,
        f"{prefix}_valid": valid,
    }, {
        "locations": int(bp_flat.shape[1]),
        "top10_bp_k": bp_k10,
        "top10_dfa_k": dfa_k10,
        "top20_bp_k": bp_k20,
        "top20_dfa_k": dfa_k20,
    }


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator):
    values = values[np.isfinite(values)]
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 50
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def summarize(values: np.ndarray, rng: np.random.Generator):
    values = values[np.isfinite(values)]
    q25, q75 = np.quantile(values, [0.25, 0.75])
    ci_low, ci_high = bootstrap_mean_ci(values, rng)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "mean_ci95_low": float(ci_low),
        "mean_ci95_high": float(ci_high),
    }


def normalized_cam(cam: np.ndarray):
    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum == minimum:
        return np.zeros_like(cam)
    return (cam - minimum) / (maximum - minimum)


def make_figure(rows, images_file, labels, bp_model, dfa_model, mean, std, device):
    selected_indices = [int(row["test_index"]) for row in rows]
    with h5py.File(images_file, "r") as handle:
        original = np.stack([handle["x"][index] for index in selected_indices])
    tensors = torch.from_numpy(original).permute(0, 3, 1, 2).float().div_(255.0)
    tensors = (tensors - mean.cpu()) / std.cpu()
    tensors = tensors.to(device)
    bp_logits, _, _, bp_maps = grad_cam(bp_model, tensors)
    dfa_logits, _, _, dfa_maps = grad_cam(dfa_model, tensors)
    bp_probs = bp_logits.softmax(dim=1)[:, 1].cpu().numpy()
    dfa_probs = dfa_logits.softmax(dim=1)[:, 1].cpu().numpy()
    bp_maps = bp_maps.cpu().numpy()
    dfa_maps = dfa_maps.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), constrained_layout=True)
    for row_index, (row, image) in enumerate(zip(rows, original)):
        agreement = "Lower agreement" if row_index == 0 else "Higher agreement"
        true_name = "malignant" if labels[int(row["test_index"])] == 1 else "benign"
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{agreement}\nOriginal; true: {true_name}", fontsize=8)
        for column, (name, cam, probability) in enumerate(
            (("BP", bp_maps[row_index], bp_probs[row_index]),
             ("DFA", dfa_maps[row_index], dfa_probs[row_index])),
            start=1,
        ):
            axes[row_index, column].imshow(image)
            axes[row_index, column].imshow(
                normalized_cam(cam), cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0
            )
            prediction = "malignant" if probability >= 0.5 else "benign"
            axes[row_index, column].set_title(
                f"{name} Grad-CAM\n{prediction}, p={probability:.3f}", fontsize=8
            )
        axes[row_index, 0].set_ylabel(
            f"IoU10={row['image_iou_top10']:.3f}; IoU20={row['image_iou_top20']:.3f}\n"
            f"Spearman={row['image_spearman']:.3f}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    OUTPUT_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
    source_png = OUTPUT_DIR / "figures" / "pcam_gradcam_agreement.png"
    source_pdf = OUTPUT_DIR / "figures" / "pcam_gradcam_agreement.pdf"
    fig.savefig(source_png, dpi=600, bbox_inches="tight")
    fig.savefig(source_pdf, bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIGURE, bbox_inches="tight")
    plt.close(fig)
    return selected_indices


def configure_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "execution.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main():
    configure_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda")
    logging.info("Using %s", torch.cuda.get_device_name(0))

    bp_checkpoint = load_checkpoint(BP_CHECKPOINT)
    dfa_checkpoint = load_checkpoint(DFA_CHECKPOINT)
    required_keys = {
        "model_state_dict", "config", "training_method", "mean", "std",
        "class_names", "epoch", "metrics",
    }
    for name, checkpoint in (("BP", bp_checkpoint), ("DFA", dfa_checkpoint)):
        if set(checkpoint) != required_keys:
            raise RuntimeError(f"{name} checkpoint keys differ: {set(checkpoint) ^ required_keys}")
    for key in ("config", "mean", "std", "class_names"):
        if bp_checkpoint[key] != dfa_checkpoint[key]:
            raise RuntimeError(f"Checkpoint metadata mismatch for {key}")
    if bp_checkpoint["class_names"] != ["Benign", "Malignant"]:
        raise RuntimeError(f"Unexpected class encoding: {bp_checkpoint['class_names']}")
    if bp_checkpoint["training_method"] != "backpropagation":
        raise RuntimeError("Unexpected BP training method")
    if dfa_checkpoint["training_method"] != "direct_feedback_alignment":
        raise RuntimeError("Unexpected DFA training method")

    bp_model = load_model(bp_checkpoint).to(device)
    dfa_original = load_model(dfa_checkpoint).to(device)
    dfa_model = load_model(dfa_checkpoint).to(device)
    mean = torch.tensor(bp_checkpoint["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(bp_checkpoint["std"], device=device).view(1, 3, 1, 1)

    with h5py.File(TEST_LABELS, "r") as handle:
        labels = handle["y"][:].reshape(-1).astype(np.int64)
    with h5py.File(TEST_IMAGES, "r") as handle:
        image_shape = tuple(handle["x"].shape)
    if image_shape[0] != len(labels) or image_shape[1:] != (96, 96, 3):
        raise RuntimeError(f"Unexpected held-out data shape: {image_shape}, labels={labels.shape}")
    indices = np.arange(len(labels), dtype=np.int64)
    pd.DataFrame({"test_index": indices, "true_class": labels}).to_csv(
        OUTPUT_DIR / "selected_indices.csv", index=False
    )

    with h5py.File(TEST_IMAGES, "r") as handle:
        probe = torch.from_numpy(handle["x"][:8]).permute(0, 3, 1, 2).float().div_(255.0)
    probe = ((probe.to(device) - mean) / std)
    with torch.no_grad():
        original_logits = dfa_original(probe)
        copy_logits = dfa_model(probe)
    dfa_copy_max_abs_difference = float((original_logits - copy_logits).abs().max().cpu())
    if not torch.allclose(original_logits, copy_logits, rtol=1e-6, atol=1e-7):
        raise RuntimeError("DFA attribution-copy logits do not match inference logits")
    del dfa_original, original_logits, copy_logits

    rows = []
    shape_checks = None
    mask_checks = {}
    with h5py.File(TEST_IMAGES, "r") as handle:
        dataset = handle["x"]
        for start in range(0, len(labels), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(labels))
            raw_images = dataset[start:stop]
            images = torch.from_numpy(raw_images).permute(0, 3, 1, 2).float().div_(255.0)
            images = (images.to(device) - mean) / std
            bp_logits, bp_raw_native, bp_native, bp_image = grad_cam(bp_model, images)
            dfa_logits, dfa_raw_native, dfa_native, dfa_image = grad_cam(dfa_model, images)
            if bp_native.shape != dfa_native.shape or bp_image.shape != dfa_image.shape:
                raise RuntimeError("BP and DFA map dimensions differ")
            shape_checks = {
                "native": list(bp_native.shape[1:]),
                "image_space": list(bp_image.shape[1:]),
            }
            bp_native_np = bp_native.cpu().numpy()
            dfa_native_np = dfa_native.cpu().numpy()
            bp_image_np = bp_image.cpu().numpy()
            dfa_image_np = dfa_image.cpu().numpy()
            image_metrics, image_masks = map_metrics(bp_image_np, dfa_image_np, "image")
            native_metrics, native_masks = map_metrics(bp_native_np, dfa_native_np, "native")
            mask_checks = {"image_space": image_masks, "native": native_masks}
            bp_prob = bp_logits.softmax(dim=1)[:, 1].cpu().numpy()
            dfa_prob = dfa_logits.softmax(dim=1)[:, 1].cpu().numpy()
            bp_pred = bp_logits.argmax(dim=1).cpu().numpy()
            dfa_pred = dfa_logits.argmax(dim=1).cpu().numpy()
            for offset, index in enumerate(range(start, stop)):
                reasons = []
                for resolution, metrics in (("image", image_metrics), ("native", native_metrics)):
                    if not metrics[f"{resolution}_bp_finite"][offset]:
                        reasons.append(f"{resolution}_bp_nonfinite")
                    if not metrics[f"{resolution}_dfa_finite"][offset]:
                        reasons.append(f"{resolution}_dfa_nonfinite")
                    if metrics[f"{resolution}_bp_constant"][offset]:
                        reasons.append(f"{resolution}_bp_constant_after_relu")
                    if metrics[f"{resolution}_dfa_constant"][offset]:
                        reasons.append(f"{resolution}_dfa_constant_after_relu")
                row = {
                    "test_index": index,
                    "true_class": int(labels[index]),
                    "bp_positive_probability": float(bp_prob[offset]),
                    "dfa_positive_probability": float(dfa_prob[offset]),
                    "bp_prediction": int(bp_pred[offset]),
                    "dfa_prediction": int(dfa_pred[offset]),
                    "bp_correct": bool(bp_pred[offset] == labels[index]),
                    "dfa_correct": bool(dfa_pred[offset] == labels[index]),
                    "bp_raw_cam_min": float(bp_raw_native[offset].min().cpu()),
                    "bp_raw_cam_max": float(bp_raw_native[offset].max().cpu()),
                    "dfa_raw_cam_min": float(dfa_raw_native[offset].min().cpu()),
                    "dfa_raw_cam_max": float(dfa_raw_native[offset].max().cpu()),
                    "excluded": bool(reasons),
                    "exclusion_reason": ";".join(reasons),
                }
                for metrics in (image_metrics, native_metrics):
                    for key, values in metrics.items():
                        value = values[offset]
                        row[key] = bool(value) if values.dtype == bool else float(value)
                rows.append(row)
            if start % (BATCH_SIZE * 32) == 0:
                logging.info("Processed %d/%d images", stop, len(labels))
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "per_image_results.csv", index=False, float_format="%.10g")

    metric_names = [
        "image_iou_top10", "image_iou_top20", "image_spearman",
        "native_iou_top10", "native_iou_top20", "native_spearman",
    ]
    summary_rows = []
    for scope, subset in (
        ("overall", results),
        ("true_class_0", results[results.true_class == 0]),
        ("true_class_1", results[results.true_class == 1]),
    ):
        for metric_index, metric in enumerate(metric_names):
            rng = np.random.default_rng(BOOTSTRAP_SEED + metric_index)
            summary_rows.append({"scope": scope, "metric": metric, **summarize(subset[metric].to_numpy(), rng)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "aggregate_results.csv", index=False, float_format="%.10g")
    summary[summary.scope != "overall"].to_csv(
        OUTPUT_DIR / "class_stratified_results.csv", index=False, float_format="%.10g"
    )
    summary[summary.metric.str.startswith("native_")].to_csv(
        OUTPUT_DIR / "robustness_results.csv", index=False, float_format="%.10g"
    )
    summary.assign(checkpoint_pair="config_1", matched_training_seed="not_recorded_in_checkpoint").to_csv(
        OUTPUT_DIR / "seed_level_results.csv", index=False, float_format="%.10g"
    )

    bp_correct = results.bp_correct.to_numpy()
    dfa_correct = results.dfa_correct.to_numpy()
    performance = {
        "n_images": len(results),
        "class_0": int((labels == 0).sum()),
        "class_1": int((labels == 1).sum()),
        "bp_accuracy": float(bp_correct.mean()),
        "bp_macro_f1": float(f1_score(labels, results.bp_prediction, average="macro")),
        "bp_roc_auc": float(roc_auc_score(labels, results.bp_positive_probability)),
        "dfa_accuracy": float(dfa_correct.mean()),
        "dfa_macro_f1": float(f1_score(labels, results.dfa_prediction, average="macro")),
        "dfa_roc_auc": float(roc_auc_score(labels, results.dfa_positive_probability)),
        "both_correct": int((bp_correct & dfa_correct).sum()),
        "bp_only_correct": int((bp_correct & ~dfa_correct).sum()),
        "dfa_only_correct": int((~bp_correct & dfa_correct).sum()),
        "both_incorrect": int((~bp_correct & ~dfa_correct).sum()),
    }
    with (OUTPUT_DIR / "performance_results.json").open("w", encoding="utf-8") as handle:
        json.dump(performance, handle, indent=2)

    eligible = results.dropna(subset=["image_iou_top10", "image_iou_top20", "image_spearman"]).copy()
    eligible["agreement_score"] = (
        eligible.image_iou_top10
        + eligible.image_iou_top20
        + (eligible.image_spearman + 1.0) / 2.0
    ) / 3.0
    quantiles = eligible.agreement_score.quantile([0.10, 0.90]).to_numpy()
    representatives = []
    for quantile, value in zip((0.10, 0.90), quantiles):
        candidates = eligible.assign(distance=(eligible.agreement_score - value).abs())
        selected = candidates.sort_values(["distance", "test_index"], kind="stable").iloc[0].to_dict()
        selected["selection_quantile"] = quantile
        selected["target_agreement_score"] = float(value)
        representatives.append(selected)
    pd.DataFrame(representatives).to_csv(
        OUTPUT_DIR / "representative_examples.csv", index=False, float_format="%.10g"
    )
    selected_indices = make_figure(
        representatives, TEST_IMAGES, labels, bp_model, dfa_model, mean, std, device
    )

    invalid_counts = {
        "total_images": len(results),
        "excluded_any_resolution": int(results.excluded.sum()),
        "image_bp_constant": int(results.image_bp_constant.sum()),
        "image_dfa_constant": int(results.image_dfa_constant.sum()),
        "native_bp_constant": int(results.native_bp_constant.sum()),
        "native_dfa_constant": int(results.native_dfa_constant.sum()),
        "image_bp_nonfinite": int((~results.image_bp_finite).sum()),
        "image_dfa_nonfinite": int((~results.image_dfa_finite).sum()),
        "native_bp_nonfinite": int((~results.native_bp_finite).sum()),
        "native_dfa_nonfinite": int((~results.native_dfa_finite).sum()),
    }
    validation = {
        "bp_dfa_map_dimensions_identical": True,
        "map_dimensions": shape_checks,
        "target_definitions_identical": True,
        "target_class_index": TARGET_CLASS_INDEX,
        "target_class_name": bp_checkpoint["class_names"][TARGET_CLASS_INDEX],
        "target_layer_name": TARGET_LAYER_NAME,
        "mask_checks": mask_checks,
        "iou_bounded_0_1": bool(results[["image_iou_top10", "image_iou_top20", "native_iou_top10", "native_iou_top20"]].stack().between(0, 1).all()),
        "spearman_bounded_minus1_1": bool(results[["image_spearman", "native_spearman"]].stack().between(-1, 1).all()),
        "dfa_attribution_copy_max_abs_logit_difference": dfa_copy_max_abs_difference,
        "selected_indices": selected_indices,
        "selected_sample_is_complete_test_set": True,
        "invalid_and_excluded_maps": invalid_counts,
    }
    with (OUTPUT_DIR / "validation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)

    config = {
        "analysis_seed": SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "sample_protocol": "complete held-out PCam test partition in stored order",
        "interpolation": "torch bilinear, align_corners=False",
        "gradcam_relu": True,
        "top_k_rule": "ceil(proportion * location_count), descending stable sort; original flat index breaks ties",
        "confidence_interval": "95% percentile bootstrap interval for the mean, resampled over images",
        "pooling": "one matched config_1 checkpoint pair; no pooling across training seeds",
        "training_seed": "not recorded in checkpoint metadata",
        "target_layer": TARGET_LAYER_NAME,
        "target_class_index": TARGET_CLASS_INDEX,
        "target_class_name": bp_checkpoint["class_names"][TARGET_CLASS_INDEX],
        "normalization_mean": bp_checkpoint["mean"],
        "normalization_std": bp_checkpoint["std"],
        "evaluation_augmentation": "none",
        "cuda_reproducibility": "fixed seeds and cuBLAS workspace; cudnn deterministic; benchmark disabled",
        "checkpoint_metadata": {
            "bp": {key: value for key, value in bp_checkpoint.items() if key != "model_state_dict"},
            "dfa": {key: value for key, value in dfa_checkpoint.items() if key != "model_state_dict"},
        },
        "sha256": {
            "bp_checkpoint": sha256(BP_CHECKPOINT),
            "dfa_checkpoint": sha256(DFA_CHECKPOINT),
            "test_images": sha256(TEST_IMAGES),
            "test_labels": sha256(TEST_LABELS),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    with (OUTPUT_DIR / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=lambda value: value.item())
    with (OUTPUT_DIR / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    logging.info("Completed attribution analysis for all %d held-out images", len(results))


if __name__ == "__main__":
    main()
