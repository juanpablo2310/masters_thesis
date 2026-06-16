"""
Shared embedding space with class masks for heterogeneous federated learning.

Each client trains a model with `total_classes` outputs but only computes loss
(and receives gradients) on the class indices it actually knows. This allows
a single global model to be shared across institutions with disjoint label sets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class SharedClassSpace:
    """
    Defines the global class label space shared across all federated clients.

    Example for 2 clients:
        space = SharedClassSpace({
            "client1": list(range(0, 6)),   # UNAL: classes 0–5
            "client2": list(range(6, 17)),  # Melbourne: classes 6–16
        })
    """
    client_known_classes: Dict[str, List[int]]

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


class GradientMasker:
    """
    Zeros gradients for unknown class indices in the YOLOv8 detection head.

    Hooks are registered on the final classification conv layers (cv3) of the
    Detect module. This prevents the global model from being updated for classes
    that a given client has never seen.

    Usage:
        masker = GradientMasker(yolo_model, unknown_indices=[6, 7, ..., 16])
        # ... run training ...
        masker.remove()
    """

    def __init__(self, model, unknown_indices: List[int]):
        self.unknown_indices = unknown_indices
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        if unknown_indices:
            self._register_hooks(model)

    def _register_hooks(self, model):
        detect = model.model[-1]  # YOLOv8 Detect layer
        if not hasattr(detect, 'cv3'):
            logger.warning("Model does not have a 'cv3' attribute — gradient masking skipped.")
            return

        for cv3 in detect.cv3:
            children = list(cv3.children())
            # The last child in cv3 is the Conv that outputs nc channels
            last = children[-1]
            # Unwrap one more level if it's a Sequential/Conv wrapper
            if hasattr(last, 'conv'):
                last = last.conv

            if hasattr(last, 'weight') and last.weight is not None:
                h = last.weight.register_hook(self._make_weight_hook())
                self._hooks.append(h)
            if hasattr(last, 'bias') and last.bias is not None:
                h = last.bias.register_hook(self._make_bias_hook())
                self._hooks.append(h)

        logger.info(
            f"GradientMasker: registered {len(self._hooks)} hooks, "
            f"masking {len(self.unknown_indices)} unknown class indices."
        )

    def _make_weight_hook(self):
        unknown = self.unknown_indices

        def hook(grad: torch.Tensor) -> torch.Tensor:
            # weight shape: [nc, in_channels, kH, kW]
            if grad is None:
                return grad
            masked = grad.clone()
            masked[unknown] = 0.0
            return masked

        return hook

    def _make_bias_hook(self):
        unknown = self.unknown_indices

        def hook(grad: torch.Tensor) -> torch.Tensor:
            # bias shape: [nc]
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


def make_label_remap_callback(class_offset: int):
    """
    Returns a YOLO callback that remaps local (0-based) class labels to
    their global index by adding `class_offset` to every label in the batch.

    Register with:
        model.add_callback("on_train_batch_start", make_label_remap_callback(offset))
    """
    def remap_labels(trainer):
        if trainer.batch is not None and 'cls' in trainer.batch:
            trainer.batch['cls'] = trainer.batch['cls'] + class_offset

    return remap_labels
