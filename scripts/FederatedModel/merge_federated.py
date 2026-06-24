"""
Option A — Head-splice merge for the shared-embedding federated model.

Each client only trains its own slice of the 17-class detection head (UNAL writes
channels 0–5, Melbourne writes channels 6–16; the rest stay at initialization).
No single trained checkpoint therefore has both halves alive.

This module builds a *real* federated 17-class model by:
  - splicing the per-class output conv of the detection head (cv3 final conv):
    rows 0–5 taken from the UNAL checkpoint, rows 6–16 from the Melbourne one;
  - FedAvg-averaging the shared trunk (backbone + neck + box head + cv3 feature
    convs) that both clients trained jointly.

Both client checkpoints must be 17-class models from the shared space (same
architecture), ideally from the same round so their trunks are compatible.

CLI:
    python merge_federated.py \
        --unal  ".../runs/train401/weights/best.pt" \
        --melu  ".../runs/train402/weights/best.pt" \
        --out   ".../Federated/merged_best.pt"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_ckpt(path: Union[str, Path]):
    """Return (full_checkpoint, model_module, state_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return ck, model, model.state_dict()


def detect_trained_group(path: Union[str, Path], unal_idx=range(0, 6), melu_idx=range(6, 17)):
    """Heuristic: which head slice was actually trained.

    Untrained class channels keep an identical initialization bias, so the slice
    with higher bias variance is the trained one. Returns 'UNAL' / 'MELU' / info.
    """
    _, _, sd = _load_ckpt(path)
    bkeys = [k for k in sd if ".cv3." in k and k.endswith(".2.bias")]
    if not bkeys:
        return {"nc": None, "trained": "unknown"}
    nc = sd[bkeys[0]].shape[0]
    bias = np.zeros(nc)
    for k in bkeys:
        bias += sd[k].numpy()
    bias /= len(bkeys)
    if nc != 17:
        return {"nc": nc, "trained": "n/a (modelo local, no 17-clases)"}
    us = float(bias[list(unal_idx)].std())
    ms = float(bias[list(melu_idx)].std())
    return {"nc": 17, "trained": "UNAL" if us > ms else "MELU",
            "std_unal": round(us, 5), "std_melu": round(ms, 5)}


def merge_federated_model(
    unal_ckpt: Union[str, Path],
    melu_ckpt: Union[str, Path],
    output_path: Union[str, Path],
    global_names: Dict[int, str],
    unal_indices: Sequence[int] = range(0, 6),
    melu_indices: Sequence[int] = range(6, 17),
    trunk: str = "fedavg",
    trunk_weights: Optional[Sequence[float]] = None,
) -> Path:
    """Splice the two client heads onto a shared trunk and save a 17-class model.

    Args:
        unal_ckpt: 17-class checkpoint with channels 0–5 trained (UNAL run).
        melu_ckpt: 17-class checkpoint with channels 6–16 trained (Melbourne run).
        output_path: where to write the merged checkpoint.
        global_names: {global_index: class_name} for all 17 classes.
        unal_indices / melu_indices: head rows owned by each client.
        trunk: how to combine the shared trunk — 'fedavg' (weighted average),
               'melu' (keep Melbourne trunk) or 'unal' (keep UNAL trunk).
        trunk_weights: (w_unal, w_melu) for 'fedavg'; defaults to equal weights.
    """
    unal_indices = list(unal_indices)
    melu_indices = list(melu_indices)
    nc = len(global_names)

    ck_mel, model_mel, sd_mel = _load_ckpt(melu_ckpt)
    _, _, sd_un = _load_ckpt(unal_ckpt)

    if trunk_weights is None:
        w_un, w_mel = 0.5, 0.5
    else:
        w_un, w_mel = trunk_weights
        s = float(w_un + w_mel)
        w_un, w_mel = w_un / s, w_mel / s

    # Per-class output convs of the detection head (cv3 final conv): shape[0] == nc.
    head_keys = {
        k for k in sd_mel
        if ".cv3." in k and sd_mel[k].ndim >= 1 and sd_mel[k].shape[0] == nc
    }

    merged: Dict[str, torch.Tensor] = {}
    n_avg = n_keep = 0
    for k, v_mel in sd_mel.items():
        v_un = sd_un.get(k)

        if k in head_keys:
            # Splice per-class rows: UNAL owns its indices, Melbourne owns the rest.
            out = v_mel.clone()
            if v_un is not None:
                idx = torch.tensor(unal_indices, dtype=torch.long)
                out[idx] = v_un.index_select(0, idx).to(v_mel.dtype)
            merged[k] = out
            continue

        if v_un is None or v_un.shape != v_mel.shape or not v_mel.is_floating_point():
            merged[k] = v_mel        # BN counters, mismatches → keep Melbourne's
            n_keep += 1
        elif trunk == "melu":
            merged[k] = v_mel
            n_keep += 1
        elif trunk == "unal":
            merged[k] = v_un
            n_keep += 1
        else:  # fedavg
            merged[k] = (w_un * v_un.float() + w_mel * v_mel.float()).to(v_mel.dtype)
            n_avg += 1

    missing, unexpected = model_mel.load_state_dict(merged, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected")

    # Make the merged model self-describing and the canonical one in the checkpoint.
    model_mel.names = {i: global_names[i] for i in range(nc)}
    if hasattr(model_mel, "nc"):
        model_mel.nc = nc
    if isinstance(ck_mel, dict):
        ck_mel["model"] = model_mel
        ck_mel["ema"] = None  # force ultralytics to use 'model' (our merged weights)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck_mel, output_path)

    logger.info(
        f"Merged model saved to {output_path}\n"
        f"  head slices: UNAL rows {unal_indices} from {Path(unal_ckpt).parent.parent.name}, "
        f"MELU rows {melu_indices} from {Path(melu_ckpt).parent.parent.name}\n"
        f"  trunk: {trunk} (w_unal={w_un:.3f}, w_melu={w_mel:.3f}) — "
        f"{n_avg} tensors averaged, {n_keep} kept, {len(head_keys)} head tensors spliced"
    )
    return output_path


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _default_global_names() -> Dict[int, str]:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
    from scripts.utils.paths import get_project_configs
    from class_mask import SharedClassSpace
    import yaml

    def load_names(p):
        cfg = yaml.safe_load(open(get_project_configs(p)))
        names = cfg.get("names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        return {int(k): str(v) for k, v in names.items()}

    space = SharedClassSpace({"client1": list(range(0, 6)), "client2": list(range(6, 17))})
    space.assign_names({
        "client1": load_names("yaml/config_un.yaml"),
        "client2": load_names("yaml/config_melu.yaml"),
    })
    return space.get_global_names()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Empalme de cabezas (Opción A): construye un modelo federado real de 17 clases."
    )
    parser.add_argument("--unal", required=True, help="Checkpoint 17-clases con canales 0–5 entrenados (run UNAL).")
    parser.add_argument("--melu", required=True, help="Checkpoint 17-clases con canales 6–16 entrenados (run Melbourne).")
    parser.add_argument("--out", required=True, help="Ruta de salida del modelo federado fusionado.")
    parser.add_argument("--trunk", default="fedavg", choices=["fedavg", "melu", "unal"],
                        help="Cómo combinar el tronco compartido.")
    parser.add_argument("--w-unal", type=float, default=None, help="Peso del tronco UNAL (fedavg).")
    parser.add_argument("--w-melu", type=float, default=None, help="Peso del tronco Melbourne (fedavg).")
    args = parser.parse_args()

    print("Grupos entrenados detectados:")
    print(f"  UNAL ckpt: {detect_trained_group(args.unal)}")
    print(f"  MELU ckpt: {detect_trained_group(args.melu)}")

    tw = None
    if args.w_unal is not None and args.w_melu is not None:
        tw = (args.w_unal, args.w_melu)

    merge_federated_model(
        unal_ckpt=args.unal,
        melu_ckpt=args.melu,
        output_path=args.out,
        global_names=_default_global_names(),
        trunk=args.trunk,
        trunk_weights=tw,
    )


if __name__ == "__main__":
    main()
