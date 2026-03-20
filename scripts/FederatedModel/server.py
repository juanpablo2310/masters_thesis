from abc import ABC, abstractmethod
import logging
from ultralytics import YOLO
from typing import List, Dict
import torch
import numpy as np

from client import FederatedClient, EnhancedFederatedClient
from metrics import MetricsTracker
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
        
    def calculate_weights(self) -> List[float]:
        """Calculate client weights based on recent performance"""
        weights = []
        for client_id in range(self.num_clients):
            history = self.performance_history[str(client_id)]
            if not history:
                weights.append(1.0)
                continue
                
            # Use recent mAP as weight
            recent_map = history[-1].get('mAP', 0)
            weights.append(max(0.1, recent_map))  # Ensure minimum weight
            
        # Normalize weights
        total = sum(weights)
        return [w/total for w in weights]



class FederatedServer:
    """Central server for federated learning coordination"""
    def __init__(self, 
                 aggregator: FederatedAggregator,
                 model_save_path: str = "/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/"):
        self.aggregator = aggregator
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        self.global_model = YOLO('yolov8n.pt')
        self.round = 0
        
    def aggregate_models(self, 
                        clients: List[FederatedClient], 
                        client_weights: List[float] = None):
        """Aggregate client models using the selected strategy"""
        if client_weights is None:
            client_weights = [1.0] * len(clients)
            
        client_models = [client.get_model_params() for client in clients]
        aggregated_params = self.aggregator.aggregate(client_models, client_weights)
        
        # Update global model, ensuring shapes align or recomputing if possible
        with torch.no_grad():
            for name, param in self.global_model.model.named_parameters():
                if name not in aggregated_params:
                    logger.warning(f"Aggregated parameters missing '{name}'; keeping global value")
                    continue
                agg = aggregated_params[name]
                if agg.shape != param.data.shape:
                    # attempt to re-aggregate using only clients with matching shape
                    matching_idxs = [i for i, m in enumerate(client_models)
                                     if name in m and tuple(m[name].shape) == tuple(param.data.shape)]
                    if matching_idxs:
                        logger.warning(
                            f"Shape mismatch for '{name}' (global {tuple(param.data.shape)} vs agg {tuple(agg.shape)}); "
                            "recomputing using {len(matching_idxs)} compatible clients"
                        )
                        wts = [client_weights[i] for i in matching_idxs]
                        total_w = float(sum(wts))
                        if total_w == 0:
                            norm_w = [1.0/len(wts)] * len(wts)
                        else:
                            norm_w = [w/total_w for w in wts]
                        agg = sum(norm * client_models[i][name] for norm, i in zip(norm_w, matching_idxs))
                    else:
                        logger.warning(
                            f"Cannot update '{name}': no client has matching shape {tuple(param.data.shape)}. Skipping."
                        )
                        continue
                param.data.copy_(agg)
        
        self.round += 1
        self._save_model()
        
    def _save_model(self):
        """Save the current global model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_save_path / f"global_model_round_{self.round}_{timestamp}.pt"
        torch.save(self.global_model.model.state_dict(), save_path)
        logger.info(f"Saved global model to {save_path}")
        
    def distribute_model(self, clients: List[FederatedClient]):
        """Distribute global model to all clients, skipping incompatible layers"""
        global_params = {name: param.data.clone() 
                        for name, param in self.global_model.model.named_parameters()}
        for client in clients:
            compatible = {}
            for name, tensor in global_params.items():
                # only copy if client has same shape
                client_param = client.model.model.named_parameters()
                # we can't iterate easily here so catch in try
                try:
                    local = dict(client.model.model.named_parameters())[name]
                    if local.data.shape == tensor.shape:
                        compatible[name] = tensor
                    else:
                        logger.warning(
                            f"Skipping layer '{name}' for client {client.client_id}: "
                            f"global shape {tuple(tensor.shape)} != local {tuple(local.data.shape)}"
                        )
                except KeyError:
                    logger.warning(f"Client {client.client_id} missing parameter '{name}', skipping")
            client.update_model_params(compatible)
            
            
            
class EnhancedFederatedServer(FederatedServer):
    """Enhanced server with metrics tracking and dynamic weighting"""
    def __init__(self, 
                 aggregator: FederatedAggregator,
                 model_save_path: str = "/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/federated_models",
                 metrics_tracker: MetricsTracker = None,
                 dynamic_weighting: DynamicWeighting = None):
        super().__init__(aggregator, model_save_path)
        self.metrics_tracker = metrics_tracker or MetricsTracker()
        self.dynamic_weighting = dynamic_weighting
        
    def aggregate_models(self, clients: List[EnhancedFederatedClient]):
        """Aggregate with dynamic weights and track metrics"""
        # Get client weights
        if self.dynamic_weighting:
            client_weights = self.dynamic_weighting.calculate_weights()
        else:
            client_weights = [1.0] * len(clients)
            
        # Aggregate models
        super().aggregate_models(clients, client_weights)
        
        # Evaluate and track metrics
        global_metrics = self._evaluate_global_model()
        self.metrics_tracker.add_metric(
            type(self.aggregator).__name__,
            self.round,
            global_metrics
        )
        
        # Update client performance history
        if self.dynamic_weighting:
            for client in clients:
                metrics = client.evaluate()
                self.dynamic_weighting.update_performance(client.client_id, metrics)
                
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance"""
        try:
            results = self.global_model.val()
            return {
                'mAP': results.box.map,
                'precision': results.box.p,
                'recall': results.box.r
            }
        except Exception as e:
            logger.error(f"Global model evaluation failed: {str(e)}")
            return {'mAP': 0.0, 'precision': 0.0, 'recall': 0.0}