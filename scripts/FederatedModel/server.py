from abc import ABC, abstractmethod
import logging
from ultralytics import YOLO
from typing import List, Dict
import torch
import numpy as np

from client import FederatedClient, EnhancedFederatedClient
from metrics import MetricsTracker
from class_mask import build_shared_yolo
from collections import defaultdict
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedAggregator(ABC):
    """Abstract base class for federation strategies"""
    @abstractmethod
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        pass


def _compatible_indices_and_weights(client_models: List[Dict[str, torch.Tensor]],
                                    client_weights: List[float],
                                    key: str):
    """Return indices of clients whose tensor for `key` matches the most common shape
    and the corresponding weights (not normalized). If no client has the key, returns ([],[]).
    """
    idx_shapes = []
    for i, m in enumerate(client_models):
        if key in m:
            idx_shapes.append((i, tuple(m[key].shape)))

    if not idx_shapes:
        return [], []

    shape_groups = {}
    for i, s in idx_shapes:
        shape_groups.setdefault(s, []).append(i)

    # choose the most common (mode) shape
    mode_shape, indices = max(shape_groups.items(), key=lambda kv: len(kv[1]))
    weights = [client_weights[i] for i in indices]
    return indices, weights


class FedAvg(FederatedAggregator):
    """Standard FedAvg aggregation strategy (per-key permissive)"""
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        aggregated_model = {}

        # iterate over union of keys present in any client
        all_keys = set().union(*[set(m.keys()) for m in client_models])

        for key in all_keys:
            indices, weights = _compatible_indices_and_weights(client_models, client_weights, key)
            if not indices:
                raise RuntimeError(f"No client provides parameter '{key}'")

            total_w = float(sum(weights))
            if total_w == 0:
                norm_weights = [1.0 / len(weights)] * len(weights)
            else:
                norm_weights = [w / total_w for w in weights]

            agg = None
            for nw, idx in zip(norm_weights, indices):
                tensor = client_models[idx][key]
                if agg is None:
                    agg = nw * tensor
                else:
                    agg = agg + nw * tensor

            aggregated_model[key] = agg

        return aggregated_model


class FedMedian(FederatedAggregator):
    """Median-based aggregation strategy (per-key permissive)"""
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        aggregated_model = {}
        all_keys = set().union(*[set(m.keys()) for m in client_models])

        for key in all_keys:
            indices, _ = _compatible_indices_and_weights(client_models, client_weights, key)
            if not indices:
                raise RuntimeError(f"No client provides parameter '{key}'")

            tensors = [client_models[i][key] for i in indices]
            stacked = torch.stack(tensors)
            aggregated_model[key] = torch.median(stacked, dim=0)[0]

        return aggregated_model


class FedTrimmedMean(FederatedAggregator):
    """Trimmed mean aggregation strategy (per-key permissive)"""
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio
        
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        aggregated_model = {}
        all_keys = set().union(*[set(m.keys()) for m in client_models])

        for key in all_keys:
            indices, _ = _compatible_indices_and_weights(client_models, client_weights, key)
            if not indices:
                raise RuntimeError(f"No client provides parameter '{key}'")

            tensors = [client_models[i][key] for i in indices]
            stacked = torch.stack(tensors)
            sorted_params, _ = torch.sort(stacked, dim=0)
            n_clients_key = stacked.size(0)
            n_trim_key = int(n_clients_key * self.trim_ratio)
            trimmed = sorted_params[n_trim_key:-n_trim_key] if n_trim_key > 0 else sorted_params
            aggregated_model[key] = torch.mean(trimmed, dim=0)

        return aggregated_model
    
class FedAdagrad(FederatedAggregator):
    """Adaptive gradient-based aggregation strategy (per-key permissive)"""
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accumulator = None
        
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        # lazy initialize accumulator per key with a compatible tensor
        if self.accumulator is None:
            self.accumulator = {}

        aggregated_model = {}

        all_keys = set().union(*[set(m.keys()) for m in client_models])

        for key in all_keys:
            indices, weights = _compatible_indices_and_weights(client_models, client_weights, key)
            if not indices:
                raise RuntimeError(f"No client provides parameter '{key}'")

            # normalize weights among compatible clients
            total_w = float(sum(weights))
            if total_w == 0:
                norm_weights = [1.0 / len(weights)] * len(weights)
            else:
                norm_weights = [w / total_w for w in weights]

            # pick a tensor to initialize accumulator if needed
            ref_tensor = client_models[indices[0]][key]
            if key not in self.accumulator:
                self.accumulator[key] = torch.zeros_like(ref_tensor)

            weighted_grads = None
            for nw, idx in zip(norm_weights, indices):
                t = client_models[idx][key]
                if weighted_grads is None:
                    weighted_grads = nw * t
                else:
                    weighted_grads = weighted_grads + nw * t

            self.accumulator[key] += weighted_grads.pow(2)
            adjusted_lr = self.learning_rate / (torch.sqrt(self.accumulator[key]) + self.epsilon)
            aggregated_model[key] = weighted_grads * adjusted_lr

        return aggregated_model

class FedProx(FederatedAggregator):
    """FedProx aggregation with proximal term (per-key permissive)"""
    def __init__(self, mu: float = 0.01):
        self.mu = mu
        self.global_model = None
        
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        aggregated_model = {}

        all_keys = set().union(*[set(m.keys()) for m in client_models])

        if self.global_model is None:
            # initialize a copy of a compatible global model
            self.global_model = {k: v.clone() for k, v in client_models[0].items()}

        for key in all_keys:
            indices, weights = _compatible_indices_and_weights(client_models, client_weights, key)
            if not indices:
                raise RuntimeError(f"No client provides parameter '{key}'")

            total_w = float(sum(weights))
            if total_w == 0:
                norm_weights = [1.0 / len(weights)] * len(weights)
            else:
                norm_weights = [w / total_w for w in weights]

            avg_update = None
            for nw, idx in zip(norm_weights, indices):
                t = client_models[idx][key]
                if avg_update is None:
                    avg_update = nw * t
                else:
                    avg_update = avg_update + nw * t

            # Proximal term only if shapes match between avg_update and stored global
            if key in self.global_model and tuple(self.global_model[key].shape) == tuple(avg_update.shape):
                prox_term = self.mu * (avg_update - self.global_model[key])
                aggregated_model[key] = avg_update + prox_term
            else:
                aggregated_model[key] = avg_update

        self.global_model = aggregated_model
        return aggregated_model

class DynamicWeighting:
    """Implements dynamic client weighting based on performance"""
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.performance_history = defaultdict(list)
        
    def update_performance(self, client_id: str, metrics: Dict):
        """Update client performance history"""
        self.performance_history[client_id].append(metrics)
        
    def calculate_weights(self, client_ids: List[str] = None) -> List[float]:
        """Calculate client weights based on recent performance.

        ``client_ids`` must be the *ordered* ids matching the client list passed to
        aggregation (fixes the previous bug where lookups used range(n) keys that
        never matched the stored "clientX" ids, silently yielding equal weights).
        """
        ids = client_ids if client_ids is not None else list(self.performance_history.keys())
        weights = []
        for cid in ids:
            history = self.performance_history.get(cid, [])
            if not history:
                weights.append(1.0)
                continue
            weights.append(max(0.1, history[-1].get('mAP', 0)))
        total = sum(weights) or 1.0
        return [w / total for w in weights]



class FederatedServer:
    """Central server for synchronized federated learning (shared embedding space).

    Holds a single global model with ``num_classes`` outputs and a domain-neutral
    pretrained backbone. Every round it aggregates the clients' full models
    (trunk + masked heads) and redistributes them, so the shared trunk stays in
    one loss basin and a genuinely shared representation emerges.
    """
    def __init__(self,
                 aggregator: FederatedAggregator,
                 num_classes: int,
                 class_names: Dict[int, str] = None,
                 model_save_path: str = "/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/federated_models"):
        self.aggregator = aggregator
        self.num_classes = num_classes
        self.class_names = class_names
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.global_model = build_shared_yolo(num_classes, class_names)
        # The canonical global weights live as a state_dict (the source of truth),
        # so that fusing a model during evaluation never corrupts them.
        self.global_state = self._transferable_state(self.global_model)
        self.round = 0

    @staticmethod
    def _transferable_state(model) -> Dict[str, torch.Tensor]:
        """Full state_dict (params + BN buffers) minus the non-float BN counters."""
        return {
            k: v.detach().clone()
            for k, v in model.model.state_dict().items()
            if not k.endswith("num_batches_tracked")
        }

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.global_state.items()}

    def make_eval_model(self):
        """Build a fresh (disposable) 17-class model holding the current global
        weights. Used for evaluation so the canonical weights are never fused."""
        model = build_shared_yolo(self.num_classes, self.class_names)
        own = model.model.state_dict()
        filtered = {k: v for k, v in self.global_state.items()
                    if k in own and own[k].shape == v.shape}
        model.model.load_state_dict(filtered, strict=False)
        return model

    def aggregate_models(self,
                         clients: List[FederatedClient],
                         client_weights: List[float] = None):
        """FedAvg-aggregate the clients' full models into the global state."""
        if client_weights is None:
            client_weights = [1.0] * len(clients)

        client_models = [client.get_model_params() for client in clients]
        aggregated = self.aggregator.aggregate(client_models, client_weights)

        ref = self.global_state
        filtered = {k: v for k, v in aggregated.items()
                    if k in ref and ref[k].shape == v.shape}
        missing = [k for k in ref if k not in filtered]
        if missing:
            logger.warning(f"aggregate: {len(missing)} global tensors not updated this round")
        new_state = {k: v.clone() for k, v in self.global_state.items()}
        new_state.update({k: v.detach().clone() for k, v in filtered.items()})
        self.global_state = new_state

        self.round += 1
        self._save_model()

    def _save_model(self):
        """Save the current global model as a full YOLO checkpoint (loadable directly)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_save_path / f"global_model_round_{self.round}_{timestamp}.pt"
        model = self.make_eval_model()
        model.model.names = self.class_names or model.model.names
        torch.save({"model": model.model, "train_args": {}}, save_path)
        self.last_save_path = save_path
        logger.info(f"Saved global model to {save_path}")

    def distribute_model(self, clients: List[FederatedClient]):
        """Push the global weights (params + BN buffers) to every client."""
        state = self.get_global_state()
        for client in clients:
            client.update_model_params(state)


class EnhancedFederatedServer(FederatedServer):
    """Server with metrics tracking; evaluation is driven by the trainer (which
    owns the per-client evaluation datasets in the shared label space)."""
    def __init__(self,
                 aggregator: FederatedAggregator,
                 num_classes: int,
                 class_names: Dict[int, str] = None,
                 model_save_path: str = "/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/federated_models",
                 metrics_tracker: MetricsTracker = None,
                 dynamic_weighting: DynamicWeighting = None):
        super().__init__(aggregator, num_classes, class_names, model_save_path)
        self.metrics_tracker = metrics_tracker or MetricsTracker()
        self.dynamic_weighting = dynamic_weighting

    @staticmethod
    def val_metrics(eval_model, data_yaml: str) -> Dict[str, float]:
        """Validate a (disposable) model on a 17-class YAML and return scalar metrics."""
        try:
            results = eval_model.val(data=str(data_yaml), verbose=False)
            return {
                'mAP': float(results.box.map),
                'mAP50': float(results.box.map50),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
        except Exception as e:
            logger.error(f"Global model evaluation failed: {str(e)}")
            return {'mAP': 0.0, 'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}

    def evaluate_global_on(self, data_yaml: str) -> Dict[str, float]:
        """Validate the global weights on a 17-class dataset YAML (fresh model)."""
        return self.val_metrics(self.make_eval_model(), data_yaml)