"""
Comparison module: local (non-aggregated) models vs the federated (aggregated) model.

Produces per-class performance comparisons (precision, recall, mAP@0.5, mAP@0.5:0.95)
on each institution's validation set, and renders thesis-ready figures.

Key idea
--------
Local models live in their own label space (UNAL: 0–5, Melbourne: 0–10).
The federated model lives in the shared 17-class space (UNAL: 0–5, Melbourne: 6–16).
To evaluate the federated model on a client's data correctly, this module builds a
temporary dataset whose labels are shifted by the client's `class_offset` and whose
YAML declares all 17 global class names. Per-class metrics are then aligned back to
each institution's local class names so local vs federated bars are directly comparable.

Usage
-----
    from class_mask import SharedClassSpace
    from model_comparison import FederatedComparison

    shared_space = SharedClassSpace({
        "client1": list(range(0, 6)),
        "client2": list(range(6, 17)),
    })

    cmp = FederatedComparison(
        shared_class_space=shared_space,
        client_configs={
            "client1": "/path/config_un.yaml",
            "client2": "/path/config_melu.yaml",
        },
        client_display_names={"client1": "UNAL", "client2": "Melbourne"},
        save_dir="./comparison_results",
    )

    cmp.run(
        local_model_paths={
            "client1": "/path/unal_best.pt",
            "client2": "/path/melu_best.pt",
        },
        federated_model_path="/path/federated_best.pt",
    )
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from class_mask import SharedClassSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metric keys used throughout, with thesis-friendly Spanish labels
METRIC_KEYS = ["precision", "recall", "mAP50", "mAP50-95", "mIoU"]
METRIC_LABELS = {
    "precision": "Precisión",
    "recall": "Exhaustividad (Recall)",
    "mAP50": "mAP@0.5",
    "mAP50-95": "mAP@0.5:0.95",
    "mIoU": "IoU medio (localización)",
}

# Consistent colours for the two model families
COLOR_LOCAL = "#4C72B0"      # blue
COLOR_FED = "#DD8452"        # orange
COLOR_POS = "#55A868"        # green (federated wins)
COLOR_NEG = "#C44E52"        # red   (federated loses)


def _infer_nc_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer number of classes from a YOLOv8 detection-head state_dict.

    The classification branch (cv3) ends in a 1x1 conv whose output channels equal
    nc, stored under keys like 'model.22.cv3.<i>.2.weight' with shape (nc, C, 1, 1).
    """
    candidates = [
        int(v.shape[0])
        for k, v in state_dict.items()
        if ".cv3." in k and k.endswith(".2.weight") and getattr(v, "ndim", 0) == 4
    ]
    if not candidates:
        return None
    # most common value across the 3 detection scales
    return max(set(candidates), key=candidates.count)


def _is_bare_state_dict(obj) -> bool:
    """True if obj looks like a plain parameter state_dict (not a full checkpoint)."""
    if not isinstance(obj, dict):
        return False
    if "model" in obj and isinstance(obj["model"], torch.nn.Module):
        return False  # full ultralytics checkpoint
    values = list(obj.values())
    return len(values) > 0 and all(torch.is_tensor(v) for v in values)


def load_yolo_model(
    path: Union[str, Path],
    cfg: str = "yolov8n.yaml",
    nc: Optional[int] = None,
    names: Optional[Dict[int, str]] = None,
) -> YOLO:
    """Load a YOLO model from either a full checkpoint or a bare state_dict.

    - Full ultralytics checkpoints (best.pt / last.pt) load directly via YOLO().
    - Bare state_dicts (as saved by ``server._save_model`` →
      ``torch.save(model.state_dict(), ...)``) are rebuilt on a fresh
      DetectionModel architecture with the inferred (or provided) ``nc``.

    Args:
        path: path to the .pt file.
        cfg: architecture YAML used to rebuild bare state_dicts (default yolov8n).
        nc: override number of classes; inferred from the state_dict if None.
        names: optional {index: name} mapping assigned to the rebuilt model.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not _is_bare_state_dict(ckpt):
        # Full checkpoint — let ultralytics handle it natively.
        return YOLO(str(path))

    # Bare state_dict path.
    resolved_nc = nc or _infer_nc_from_state_dict(ckpt)
    if resolved_nc is None:
        raise ValueError(
            f"Could not infer nc from state_dict at {path}; pass nc=... explicitly."
        )
    logger.info(f"Loading bare state_dict {path.name} into a fresh nc={resolved_nc} model")

    model_obj = DetectionModel(cfg=cfg, nc=resolved_nc, verbose=False)
    missing, unexpected = model_obj.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        logger.warning(
            f"  state_dict load: {len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    model_obj.names = names or {i: f"class_{i}" for i in range(resolved_nc)}
    model_obj.args = {"imgsz": 640}

    # Wrap into a minimal ultralytics checkpoint and load through YOLO so that
    # validation / prediction utilities are fully wired up.
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        torch.save({"model": model_obj, "train_args": {}}, tmp_path)
        model = YOLO(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model


@dataclass
class PerClassResult:
    """Per-class metrics for one (model, dataset) evaluation, keyed by local class name."""
    client_id: str
    model_kind: str  # "local" or "federated"
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # metrics[class_name][metric_key] = value


class FederatedComparison:
    """Evaluate and compare local vs federated models per class, then plot."""

    def __init__(
        self,
        shared_class_space: SharedClassSpace,
        client_configs: Dict[str, Union[str, Path]],
        client_display_names: Optional[Dict[str, str]] = None,
        save_dir: Union[str, Path] = "./comparison_results",
        imgsz: int = 640,
        batch: int = 16,
        device: Optional[str] = None,
    ):
        self.space = shared_class_space
        self.client_configs = {cid: Path(p) for cid, p in client_configs.items()}
        self.display_names = client_display_names or {cid: cid for cid in client_configs}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.imgsz = imgsz
        self.batch = batch
        self.device = device

        # Load per-client local class names from their YAMLs
        self.local_names: Dict[str, Dict[int, str]] = {}
        for cid, cfg in self.client_configs.items():
            self.local_names[cid] = self._load_names(cfg)

        # Build global names: local name placed at (offset + local_idx)
        self.global_names: Dict[int, str] = {}
        for cid in self.client_configs:
            offset = self.space.get_class_offset(cid)
            for local_idx, name in self.local_names[cid].items():
                self.global_names[offset + local_idx] = name

        # Results storage: results[client_id][model_kind] = PerClassResult
        self.results: Dict[str, Dict[str, PerClassResult]] = {}

    # ------------------------------------------------------------------ #
    # Config / name helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_names(config_path: Path) -> Dict[int, str]:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", {})
        # YOLO accepts dict {idx: name} or list [name, ...]
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        return {int(k): str(v) for k, v in names.items()}

    @staticmethod
    def _resolve_split_dir(config_path: Path, split: str = "val") -> Path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        images_dir = cfg.get(split) or cfg.get("train")
        if images_dir is None:
            raise ValueError(f"No '{split}' or 'train' entry in {config_path}")
        return Path(images_dir)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _extract_per_class(self, results, index_to_name: Dict[int, str]) -> Dict[str, Dict[str, float]]:
        """Turn an ultralytics val() result into {class_name: {metric: value}}."""
        box = results.box
        out: Dict[str, Dict[str, float]] = {}
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_idx = int(cls_idx)
            name = index_to_name.get(cls_idx, f"class_{cls_idx}")
            p, r, ap50, ap = box.class_result(i)
            out[name] = {
                "precision": float(p),
                "recall": float(r),
                "mAP50": float(ap50),
                "mAP50-95": float(ap),
            }
        return out

    def _eval_local(self, client_id: str, model_path: Union[str, Path],
                    compute_iou: bool = True) -> PerClassResult:
        """Evaluate a local model on its own dataset in the local class space."""
        cfg = self.client_configs[client_id]
        local_names = self.local_names[client_id]
        logger.info(f"[{client_id}] evaluating LOCAL model: {model_path}")
        model = load_yolo_model(model_path, nc=len(local_names), names=local_names)
        results = model.val(
            data=str(cfg),
            imgsz=self.imgsz,
            batch=self.batch,
            device=self._device(),
            verbose=False,
        )
        metrics = self._extract_per_class(results, local_names)
        if compute_iou:
            miou = self._compute_miou(client_id, model, model_kind="local")
            self._merge_metric(metrics, miou, "mIoU")
        return PerClassResult(client_id=client_id, model_kind="local", metrics=metrics)

    def _eval_federated(self, client_id: str, model_path: Union[str, Path],
                        keep_temp: bool = False, compute_iou: bool = True) -> PerClassResult:
        """Evaluate the federated model on a client's data in the shared 17-class space."""
        offset = self.space.get_class_offset(client_id)
        logger.info(f"[{client_id}] evaluating FEDERATED model (offset={offset}): {model_path}")

        # Load once (handles bare state_dict from server._save_model) and reuse for IoU.
        model = load_yolo_model(
            model_path, nc=self.space.total_classes, names=self.global_names
        )

        tmp_yaml, tmp_root = self._build_global_eval_dataset(client_id, offset)
        try:
            results = model.val(
                data=str(tmp_yaml),
                imgsz=self.imgsz,
                batch=self.batch,
                device=self._device(),
                verbose=False,
            )
            # global index -> local name (only this client's indices appear in its data)
            metrics = self._extract_per_class(results, self.global_names)
        finally:
            if not keep_temp:
                shutil.rmtree(tmp_root, ignore_errors=True)

        if compute_iou:
            miou = self._compute_miou(client_id, model, model_kind="federated")
            self._merge_metric(metrics, miou, "mIoU")

        return PerClassResult(client_id=client_id, model_kind="federated", metrics=metrics)

    # ------------------------------------------------------------------ #
    # Intersection-over-Union (localization quality) per class
    # ------------------------------------------------------------------ #
    def _compute_miou(
        self,
        client_id: str,
        model: YOLO,
        model_kind: str,
        conf: float = 0.25,
        iou_thr: float = 0.5,
        max_images: Optional[int] = None,
    ) -> Dict[str, float]:
        """Mean IoU of matched (TP) detections per class, in the local class space.

        Runs prediction over the validation images, greedily matches predictions to
        ground-truth boxes of the same class by IoU, and averages the IoU of matches
        whose overlap is at least ``iou_thr``. For the federated model, predicted
        global indices are shifted back to local indices by the client's offset.
        """
        from ultralytics.utils.metrics import box_iou

        offset = self.space.get_class_offset(client_id) if model_kind == "federated" else 0
        local_names = self.local_names[client_id]
        n_local = len(local_names)

        images_dir = self._resolve_split_dir(self.client_configs[client_id], "val")
        if not images_dir.exists():
            logger.warning(f"[{client_id}] images dir missing, skipping mIoU: {images_dir}")
            return {}
        labels_dir = Path(str(images_dir).replace("images", "labels"))

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
        if max_images:
            images = images[:max_images]

        per_class: Dict[str, List[float]] = defaultdict(list)
        gt_seen: set = set()

        for img in images:
            lbl = labels_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            res = model.predict(str(img), conf=conf, device=self._device(), verbose=False)[0]
            h, w = res.orig_shape

            # Ground-truth boxes (local class space)
            gts: Dict[int, List[List[float]]] = defaultdict(list)
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                gts[c].append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                               (cx + bw / 2) * w, (cy + bh / 2) * h])
                gt_seen.add(c)

            if res.boxes is None or len(res.boxes) == 0:
                continue
            pred_cls = [int(c) - offset for c in res.boxes.cls.int().tolist()]
            pred_xyxy = res.boxes.xyxy.float().cpu()

            for c in range(n_local):
                if c not in gts:
                    continue
                pred_idx = [i for i, pc in enumerate(pred_cls) if pc == c]
                if not pred_idx:
                    continue
                gt_t = torch.tensor(gts[c], dtype=torch.float32)
                pr_t = pred_xyxy[pred_idx]
                iou = box_iou(gt_t, pr_t)  # [n_gt, n_pred]
                per_class[local_names[c]].extend(self._greedy_match_iou(iou, iou_thr))

        # Ensure every class with ground truth is reported (0.0 if no TP matches)
        result = {local_names[c]: 0.0 for c in gt_seen if 0 <= c < n_local}
        for name, ious in per_class.items():
            result[name] = float(np.mean(ious)) if ious else 0.0
        return result

    @staticmethod
    def _greedy_match_iou(iou: "torch.Tensor", thr: float) -> List[float]:
        """Greedy 1-to-1 matching: repeatedly take the highest remaining IoU >= thr."""
        iou = iou.clone()
        matched: List[float] = []
        while iou.numel() and float(iou.max()) >= thr:
            best = float(iou.max())
            pos = (iou == iou.max()).nonzero()[0]
            g, p = int(pos[0]), int(pos[1])
            matched.append(best)
            iou[g, :] = -1.0
            iou[:, p] = -1.0
        return matched

    @staticmethod
    def _merge_metric(metrics: Dict[str, Dict[str, float]],
                      extra: Dict[str, float], key: str):
        """Insert an extra per-class metric (e.g. mIoU) into the metrics dict."""
        for cls_name, value in extra.items():
            metrics.setdefault(cls_name, {})[key] = float(value)

    def _build_global_eval_dataset(self, client_id: str, offset: int):
        """
        Build a temporary YOLO dataset whose labels are shifted by `offset` and whose
        YAML declares all global class names. Images are symlinked; labels are rewritten.
        Returns (yaml_path, tmp_root).
        """
        images_dir = self._resolve_split_dir(self.client_configs[client_id], "val")
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Validation images dir not found for {client_id}: {images_dir}. "
                "Is the external drive mounted?"
            )
        labels_dir = Path(str(images_dir).replace("images", "labels"))

        tmp_root = Path(tempfile.mkdtemp(prefix=f"fedcmp_{client_id}_"))
        tmp_images = tmp_root / "images"
        tmp_labels = tmp_root / "labels"
        tmp_images.mkdir(parents=True, exist_ok=True)
        tmp_labels.mkdir(parents=True, exist_ok=True)

        # Symlink images
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for img in images_dir.iterdir():
            if img.suffix.lower() in exts:
                link = tmp_images / img.name
                try:
                    link.symlink_to(img.resolve())
                except FileExistsError:
                    pass

        # Rewrite labels with class offset
        if labels_dir.exists():
            for lbl in labels_dir.glob("*.txt"):
                shifted_lines = []
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    parts[0] = str(int(float(parts[0])) + offset)
                    shifted_lines.append(" ".join(parts))
                (tmp_labels / lbl.name).write_text("\n".join(shifted_lines) + ("\n" if shifted_lines else ""))
        else:
            logger.warning(f"Labels dir not found for {client_id}: {labels_dir}")

        # Write global YAML
        total = self.space.total_classes 
        names = {i: self.global_names.get(i, f"class_{i}") for i in range(total)}
        yaml_path = tmp_root / "global_eval.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(
                {
                    "path": str(tmp_root),
                    "val": "images",
                    "train" : "images",
                    "names": names,
                    "nc": total,
                },
                f,
                sort_keys=False,
                allow_unicode=True,
            )
        return yaml_path, tmp_root

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(
        self,
        local_model_paths: Dict[str, Union[str, Path]],
        federated_model_path: Union[str, Path],
        keep_temp: bool = False,
        compute_iou: bool = True,
    ):
        """Evaluate all models and generate every comparison artefact.

        Args:
            local_model_paths: {client_id: path} for the non-aggregated models.
            federated_model_path: path to the aggregated model (full checkpoint or
                bare state_dict, e.g. the one saved by ``server._save_model``).
            keep_temp: keep the temporary offset-remapped datasets (debug).
            compute_iou: also compute per-class mean IoU (localization quality).
        """
        for client_id in self.client_configs:
            self.results.setdefault(client_id, {})
            if client_id in local_model_paths:
                self.results[client_id]["local"] = self._eval_local(
                    client_id, local_model_paths[client_id], compute_iou=compute_iou
                )
            self.results[client_id]["federated"] = self._eval_federated(
                client_id, federated_model_path,
                keep_temp=keep_temp, compute_iou=compute_iou,
            )

        # Artefacts
        for client_id in self.results:
            self.plot_metrics_panel(client_id)
            for metric in METRIC_KEYS:
                self.plot_per_class(client_id, metric)
            self.plot_delta(client_id, "mAP50")
            if compute_iou:
                self.plot_delta(client_id, "mIoU")
        self.plot_macro_summary()
        self.export_table()
        logger.info(f"Comparison artefacts written to: {self.save_dir}")
        return self.results

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def _ordered_classes(self, client_id: str) -> List[str]:
        """Class names in this client's local index order (for stable x-axis)."""
        offset = self.space.get_class_offset(client_id)
        local = self.local_names[client_id]
        return [local[k] for k in sorted(local.keys())]

    def _values(self, client_id: str, model_kind: str, metric: str) -> Dict[str, float]:
        res = self.results.get(client_id, {}).get(model_kind)
        if res is None:
            return {}
        return {name: m.get(metric, np.nan) for name, m in res.metrics.items()}

    def _save_fig(self, fig, name: str):
        png = self.save_dir / f"{name}.png"
        pdf = self.save_dir / f"{name}.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")  # vector for thesis
        plt.close(fig)
        logger.info(f"  saved {png.name} / {pdf.name}")

    def plot_per_class(self, client_id: str, metric: str):
        """Grouped bar chart: Local vs Federated for one metric on one dataset."""
        classes = self._ordered_classes(client_id)
        local_vals = self._values(client_id, "local", metric)
        fed_vals = self._values(client_id, "federated", metric)

        local_y = [local_vals.get(c, 0.0) for c in classes]
        fed_y = [fed_vals.get(c, 0.0) for c in classes]

        x = np.arange(len(classes))
        w = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.1), 5.2))
        b1 = ax.bar(x - w / 2, local_y, w, label="Local (sin agregar)", color=COLOR_LOCAL)
        b2 = ax.bar(x + w / 2, fed_y, w, label="Federado (agregado)", color=COLOR_FED)

        self._annotate(ax, b1)
        self._annotate(ax, b2)

        disp = self.display_names.get(client_id, client_id)
        ax.set_title(f"{METRIC_LABELS[metric]} por clase — {disp}", fontsize=13, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        self._save_fig(fig, f"{client_id}_{metric}_per_class")

    def plot_metrics_panel(self, client_id: str):
        """Panel with every metric for one dataset (grid adapts to metric count)."""
        classes = self._ordered_classes(client_id)
        x = np.arange(len(classes))
        w = 0.38
        n = len(METRIC_KEYS)
        ncols = 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(max(12, len(classes) * 1.4), 4.5 * nrows)
        )
        axes = np.atleast_1d(axes).flatten()
        disp = self.display_names.get(client_id, client_id)
        fig.suptitle(f"Local vs Federado por clase — {disp}", fontsize=15, fontweight="bold")

        for ax, metric in zip(axes, METRIC_KEYS):
            local_vals = self._values(client_id, "local", metric)
            fed_vals = self._values(client_id, "federated", metric)
            local_y = [local_vals.get(c, 0.0) for c in classes]
            fed_y = [fed_vals.get(c, 0.0) for c in classes]
            ax.bar(x - w / 2, local_y, w, label="Local", color=COLOR_LOCAL)
            ax.bar(x + w / 2, fed_y, w, label="Federado", color=COLOR_FED)
            ax.set_title(METRIC_LABELS[metric], fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)

        # Hide any unused subplots
        for ax in axes[n:]:
            ax.set_visible(False)

        axes[0].legend(frameon=False, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(fig, f"{client_id}_panel_all_metrics")

    def plot_delta(self, client_id: str, metric: str):
        """Diverging bar of (Federated - Local) per class; green = fed improves."""
        classes = self._ordered_classes(client_id)
        local_vals = self._values(client_id, "local", metric)
        fed_vals = self._values(client_id, "federated", metric)
        deltas = [fed_vals.get(c, 0.0) - local_vals.get(c, 0.0) for c in classes]
        colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.1), 5.2))
        bars = ax.bar(classes, deltas, color=colors)
        for bar, d in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                d + (0.005 if d >= 0 else -0.005),
                f"{d:+.3f}",
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=8,
            )
        disp = self.display_names.get(client_id, client_id)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Δ {METRIC_LABELS[metric]} (Federado − Local) — {disp}",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylabel(f"Δ {METRIC_LABELS[metric]}")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        self._save_fig(fig, f"{client_id}_{metric}_delta")

    def plot_macro_summary(self):
        """Macro-averaged metrics (Local vs Federated) per institution."""
        clients = list(self.results.keys())
        labels = [self.display_names.get(c, c) for c in clients]
        x = np.arange(len(METRIC_KEYS))
        n = len(clients)
        group_w = 0.8
        bar_w = group_w / (2 * n)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for ci, client_id in enumerate(clients):
            local_macro = [self._macro(client_id, "local", m) for m in METRIC_KEYS]
            fed_macro = [self._macro(client_id, "federated", m) for m in METRIC_KEYS]
            off_local = (2 * ci) * bar_w - group_w / 2 + bar_w / 2
            off_fed = (2 * ci + 1) * bar_w - group_w / 2 + bar_w / 2
            ax.bar(x + off_local, local_macro, bar_w,
                   label=f"{labels[ci]} — Local", color=COLOR_LOCAL, alpha=0.6 + 0.4 * ci / max(n, 1))
            ax.bar(x + off_fed, fed_macro, bar_w,
                   label=f"{labels[ci]} — Federado", color=COLOR_FED, alpha=0.6 + 0.4 * ci / max(n, 1))

        ax.set_title("Promedio macro por institución: Local vs Federado",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_KEYS])
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        self._save_fig(fig, "macro_summary")

    def _macro(self, client_id: str, model_kind: str, metric: str) -> float:
        vals = list(self._values(client_id, model_kind, metric).values())
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _annotate(ax, bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # ------------------------------------------------------------------ #
    # Tabular export
    # ------------------------------------------------------------------ #
    def export_table(self):
        """Write a long-format CSV with every (client, class, model, metric) value."""
        import csv
        rows = []
        for client_id, kinds in self.results.items():
            disp = self.display_names.get(client_id, client_id)
            for model_kind, res in kinds.items():
                for cls_name, metrics in res.metrics.items():
                    for metric, value in metrics.items():
                        rows.append({
                            "institucion": disp,
                            "client_id": client_id,
                            "modelo": model_kind,
                            "clase": cls_name,
                            "metrica": metric,
                            "valor": round(float(value), 5),
                        })
        csv_path = self.save_dir / "comparison_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["institucion", "client_id", "modelo", "clase", "metrica", "valor"]
            )
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  saved {csv_path.name} ({len(rows)} rows)")
        return csv_path


# ---------------------------------------------------------------------- #
# CLI: default wiring for the UNAL + Melbourne herbarium setup
# ---------------------------------------------------------------------- #
def _build_default_comparison(save_dir: str) -> "FederatedComparison":
    """Construct a FederatedComparison with the project's standard 17-class setup."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
    from scripts.utils.paths import get_project_configs

    shared_space = SharedClassSpace({
        "client1": list(range(0, 6)),    # UNAL
        "client2": list(range(6, 17)),   # Melbourne
    })
    return FederatedComparison(
        shared_class_space=shared_space,
        client_configs={
            "client1": get_project_configs("yaml/config_un.yaml"),
            "client2": get_project_configs("yaml/config_melu.yaml"),
        },
        client_display_names={"client1": "UNAL", "client2": "Melbourne"},
        save_dir=save_dir,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compara modelos locales (sin agregar) vs el modelo federado (agregado) por clase."
    )
    parser.add_argument("--federated", required=True,
                        help="Ruta al checkpoint YOLO del modelo federado (17 clases).")
    parser.add_argument("--local-unal", required=True,
                        help="Ruta al checkpoint YOLO del modelo local de UNAL.")
    parser.add_argument("--local-melu", required=True,
                        help="Ruta al checkpoint YOLO del modelo local de Melbourne.")
    parser.add_argument("--save-dir", default="./comparison_results",
                        help="Directorio de salida para gráficos y CSV.")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Conservar los datasets temporales remapeados (debug).")
    parser.add_argument("--no-iou", action="store_true",
                        help="Omitir el cálculo de IoU medio por clase (más rápido).")
    args = parser.parse_args()

    cmp = _build_default_comparison(args.save_dir)
    cmp.run(
        local_model_paths={
            "client1": args.local_unal,
            "client2": args.local_melu,
        },
        federated_model_path=args.federated,
        keep_temp=args.keep_temp,
        compute_iou=not args.no_iou,
    )


if __name__ == "__main__":
    main()
