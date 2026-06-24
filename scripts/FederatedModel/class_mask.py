"""
Shared embedding space with class masks for heterogeneous federated learning.

Each client trains a model with `total_classes` outputs but only computes loss
(and receives gradients) on the class indices it actually knows. This allows
a single global model to be shared across institutions with disjoint label sets.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_shared_yolo(nc: int, names: Optional[Dict[int, str]] = None,
                      pretrained: str = "yolov8n.pt", cfg: str = "yolov8n.yaml"):
    """Build a YOLO model with `nc` outputs and a domain-neutral pretrained backbone.

    Loads the COCO-pretrained backbone/neck from `pretrained` into a fresh
    `nc`-class architecture (the detection head is randomly initialized). This is
    the neutral, shared starting point for federated training — NOT a single
    client's domain checkpoint.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    # Build everything in-memory from the YAML so the parameters are normal leaf
    # tensors (a torch.save/torch.load round-trip would mark them as inference
    # tensors, which then reject the in-place load_state_dict used to sync weights).
    model = YOLO(cfg)                                  # YOLO wrapper (nc=80 default)
    dm = DetectionModel(cfg=cfg, nc=nc, verbose=False)  # fresh nc-class model

    if pretrained:
        try:
            pre = YOLO(pretrained).model.state_dict()
            tgt = dm.state_dict()
            inter = {k: v.clone() for k, v in pre.items()
                     if k in tgt and tgt[k].shape == v.shape}
            dm.load_state_dict(inter, strict=False)
            logger.info(
                f"build_shared_yolo: transferred {len(inter)}/{len(tgt)} pretrained "
                f"tensors from {pretrained} (head left random)."
            )
        except Exception as e:
            logger.warning(f"build_shared_yolo: could not load '{pretrained}': {e}")

    dm.names = {i: (names[i] if names else f"class_{i}") for i in range(nc)}
    dm.nc = nc
    if not hasattr(dm, "args") or dm.args is None:
        dm.args = getattr(model.model, "args", {"imgsz": 640})

    model.model = dm
    return model


@dataclass
class SharedClassSpace:
    """
    Defines the global class label space shared across all federated clients.

    Example for 2 clients:
        space = SharedClassSpace({
            "client1": list(range(0, 6)),   # UNAL: classes 0–5
            "client2": list(range(6, 17)),  # Melbourne: classes 6–16
        })

    Optionally carries the human-readable global class names (global index -> name),
    used when building per-client datasets in the shared space.
    """
    client_known_classes: Dict[str, List[int]]
    class_names: Optional[Dict[int, str]] = None

    @property
    def total_classes(self) -> int:
        all_indices = [idx for indices in self.client_known_classes.values() for idx in indices]
        return max(all_indices) + 1 if all_indices else 0

    def get_known_indices(self, client_id: str) -> List[int]:
        return self.client_known_classes[client_id]

    def get_class_offset(self, client_id: str) -> int:
        """Offset to add to local (0-based) labels to reach global indices."""
        indices = self.client_known_classes[client_id]
        return indices[0] if indices else 0

    def get_unknown_indices(self, client_id: str) -> List[int]:
        known = set(self.client_known_classes[client_id])
        return [i for i in range(self.total_classes) if i not in known]

    def get_global_names(self) -> Dict[int, str]:
        if self.class_names:
            return dict(self.class_names)
        return {i: f"class_{i}" for i in range(self.total_classes)}

    def assign_names(self, client_local_names: Dict[str, Dict[int, str]]):
        """Assemble the global names from each client's local {index: name} map.

        Each client's local class ``li`` is placed at global index ``offset + li``.
        """
        names: Dict[int, str] = {}
        for client_id, local_names in client_local_names.items():
            offset = self.get_class_offset(client_id)
            for local_idx, name in local_names.items():
                names[offset + local_idx] = name
        self.class_names = names
        return names


class GradientMasker:
    """
    Zeros gradients for unknown class indices in the YOLOv8 detection head.

    Hooks are registered on the final classification conv layers (cv3) of the
    Detect module. This prevents the global model from being updated for classes
    that a given client has never seen.

    Accepts a YOLO wrapper, a DetectionModel, a DDP-wrapped model, or the raw
    nn.Module; the Detect head is located by searching the module tree.

    Usage:
        masker = GradientMasker(trainer.model, unknown_indices=[6, 7, ..., 16])
        # ... run training ...
        masker.remove()
    """

    def __init__(self, model, unknown_indices: List[int]):
        self.unknown_indices = unknown_indices
        self._hooks: List = []
        self._head_params: List = []      # cv3 final conv weight/bias tensors
        if unknown_indices:
            self._register_hooks(model)

    @staticmethod
    def head_param_list(model, nc: int = None) -> List["torch.Tensor"]:
        """The per-class output conv weight/bias tensors of the Detect head."""
        detect = GradientMasker._find_detect_module(model)
        if detect is None or not hasattr(detect, "cv3"):
            return []
        nc = nc if nc is not None else getattr(detect, "nc", None)
        params = []
        for branch in detect.cv3:
            target = None
            for sub in branch.modules():
                w = getattr(sub, "weight", None)
                if w is not None and w.ndim >= 1 and (nc is None or w.shape[0] == nc):
                    target = sub
            if target is None:
                continue
            if getattr(target, "weight", None) is not None:
                params.append(target.weight)
            if getattr(target, "bias", None) is not None:
                params.append(target.bias)
        return params

    def snapshot_unknown(self, model) -> List["torch.Tensor"]:
        """Clone the unknown-class rows of the head (the values to keep frozen)."""
        idx = self.unknown_indices
        return [p.detach()[idx].clone() for p in self.head_param_list(model)]

    def restore_unknown(self, model, snapshot: List["torch.Tensor"]):
        """Hard-reset the unknown-class rows of `model`'s head to the snapshot.

        Belt-and-suspenders with the gradient hooks: guarantees the classes this
        client does not own never change, regardless of optimizer/EMA behavior.
        """
        if not snapshot:
            return
        idx = self.unknown_indices
        with torch.no_grad():
            for p, s in zip(self.head_param_list(model), snapshot):
                p.data[idx] = s.to(p.dtype).to(p.device)

    @staticmethod
    def _find_detect_module(model):
        """Locate the YOLO Detect head from a YOLO/DetectionModel/DDP/nn.Module."""
        try:
            from ultralytics.nn.modules import Detect
        except Exception:
            Detect = None

        net = model
        if not isinstance(net, torch.nn.Module):
            net = getattr(net, "model", net)  # YOLO wrapper -> DetectionModel
        if hasattr(net, "module") and isinstance(getattr(net, "module"), torch.nn.Module):
            net = net.module  # DDP unwrap
        if not isinstance(net, torch.nn.Module):
            return None

        detect = None
        for m in net.modules():
            if Detect is not None and isinstance(m, Detect):
                detect = m
            elif hasattr(m, "cv3") and hasattr(m, "nc"):
                detect = m
        return detect

    def _register_hooks(self, model):
        detect = self._find_detect_module(model)
        if detect is None or not hasattr(detect, "cv3"):
            logger.warning("Could not locate YOLO Detect head — gradient masking skipped.")
            return

        nc = getattr(detect, "nc", None)

        for branch in detect.cv3:
            # The output conv is the (last) layer whose out-channels == nc.
            target = None
            for sub in branch.modules():
                w = getattr(sub, "weight", None)
                if w is not None and w.ndim >= 1 and (nc is None or w.shape[0] == nc):
                    target = sub
            if target is None:
                continue
            for param in (getattr(target, "weight", None), getattr(target, "bias", None)):
                # Hooks can only attach to leaf tensors that require grad. During real
                # training (callback on_train_start) they do; loaded-for-inference
                # models don't, so we skip rather than crash.
                if param is not None and param.requires_grad:
                    self._hooks.append(param.register_hook(self._make_hook()))

        if not self._hooks:
            logger.warning(
                "GradientMasker: no grad-enabled head params found "
                "(is the model in training mode?)."
            )
        logger.info(
            f"GradientMasker: registered {len(self._hooks)} hooks "
            f"(nc={nc}), masking {len(self.unknown_indices)} unknown class indices."
        )

    def _make_hook(self):
        unknown = self.unknown_indices

        def hook(grad: torch.Tensor) -> torch.Tensor:
            # Works for both weight ([nc, ...]) and bias ([nc]) — index dim 0.
            if grad is None:
                return grad
            masked = grad.clone()
            masked[unknown] = 0.0
            return masked

        return hook

    def remove(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def build_global_dataset(
    config_path,
    offset: int,
    global_names: Dict[int, str],
    total_classes: int,
    splits=("train", "val"),
    max_train_images: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Build a YOLO dataset (YAML) in the shared global class space for one client.

    - Fast path (``offset == 0`` and no ``max_train_images``): emit a YAML that
      declares all ``total_classes`` names and points at the original images/labels
      (labels already sit in the global space).
    - Otherwise: symlink images and rewrite labels with the class offset added into
      a temporary directory. If ``max_train_images`` is set, the *train* split is
      randomly subsampled to that many images (varied each round via ``seed``) to
      control the number of local SGD steps per federated round.

    Returns (yaml_path, tmp_root). ``tmp_root`` should be removed when done.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    names = {i: global_names.get(i, f"class_{i}") for i in range(total_classes)}
    tmp_root = Path(tempfile.mkdtemp(prefix="fedds_"))

    if offset == 0 and max_train_images is None:
        out = {"names": names, "nc": total_classes}
        for split in splits:
            d = cfg.get(split)
            if d:
                out[split] = str(d)
        yaml_path = tmp_root / "global.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
        return yaml_path, tmp_root

    rng = random.Random(seed)
    out = {"path": str(tmp_root), "names": names, "nc": total_classes}
    for split in splits:
        d = cfg.get(split)
        if not d:
            continue
        img_dir = Path(d)
        s_img = tmp_root / split / "images"
        s_lbl = tmp_root / split / "labels"
        s_img.mkdir(parents=True, exist_ok=True)
        s_lbl.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            logger.warning(f"Image dir not found: {img_dir}")
            out[split] = f"{split}/images"
            continue

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if split == "train" and max_train_images and len(imgs) > max_train_images:
            imgs = rng.sample(imgs, max_train_images)

        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        for img in imgs:
            try:
                (s_img / img.name).symlink_to(img.resolve())
            except FileExistsError:
                pass
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                parts[0] = str(int(float(parts[0])) + offset)
                lines.append(" ".join(parts))
            (s_lbl / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

        out[split] = f"{split}/images"

    yaml_path = tmp_root / "global.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    return yaml_path, tmp_root


def make_label_remap_callback(class_offset: int):
    """
    Returns a YOLO callback that remaps local (0-based) class labels to their
    global index by adding ``class_offset`` to every label in the batch.

    Kept as a lightweight alternative to ``build_global_dataset`` for callers that
    prefer remapping on the fly. Register with:
        model.add_callback("on_train_batch_start", make_label_remap_callback(offset))
    """
    def remap_labels(trainer):
        if getattr(trainer, "batch", None) is not None and "cls" in trainer.batch:
            trainer.batch["cls"] = trainer.batch["cls"] + class_offset

    return remap_labels
