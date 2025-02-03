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

class FedAvg(FederatedAggregator):
    """Standard FedAvg aggregation strategy"""
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        normalized_weights = np.array(client_weights) / sum(client_weights)
        aggregated_model = {}
        
        for key in client_models[0].keys():
            aggregated_model[key] = sum(
                w * model[key] for w, model in zip(normalized_weights, client_models)
            )
        return aggregated_model

class FedMedian(FederatedAggregator):
    """Median-based aggregation strategy"""
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        aggregated_model = {}
        
        for key in client_models[0].keys():
            stacked_params = torch.stack([model[key] for model in client_models])
            aggregated_model[key] = torch.median(stacked_params, dim=0)[0]
        return aggregated_model

class FedTrimmedMean(FederatedAggregator):
    """Trimmed mean aggregation strategy"""
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio
        
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        aggregated_model = {}
        n_clients = len(client_models)
        n_trim = int(n_clients * self.trim_ratio)
        
        for key in client_models[0].keys():
            stacked_params = torch.stack([model[key] for model in client_models])
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            trimmed_params = sorted_params[n_trim:-n_trim] if n_trim > 0 else sorted_params
            aggregated_model[key] = torch.mean(trimmed_params, dim=0)
        return aggregated_model
    
class FedAdagrad(FederatedAggregator):
    """Adaptive gradient-based aggregation strategy"""
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accumulator = None
        
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        if self.accumulator is None:
            self.accumulator = {k: torch.zeros_like(v) 
                              for k, v in client_models[0].items()}
        
        aggregated_model = {}
        normalized_weights = np.array(client_weights) / sum(client_weights)
        
        for key in client_models[0].keys():
            # Compute weighted gradients
            weighted_grads = sum(w * model[key] 
                               for w, model in zip(normalized_weights, client_models))
            
            # Update accumulator
            self.accumulator[key] += weighted_grads.pow(2)
            
            # Compute update
            adjusted_lr = self.learning_rate / (torch.sqrt(self.accumulator[key]) + self.epsilon)
            aggregated_model[key] = weighted_grads * adjusted_lr
            
        return aggregated_model

class FedProx(FederatedAggregator):
    """FedProx aggregation with proximal term"""
    def __init__(self, mu: float = 0.01):
        self.mu = mu
        self.global_model = None
        
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        normalized_weights = np.array(client_weights) / sum(client_weights)
        aggregated_model = {}
        
        if self.global_model is None:
            self.global_model = client_models[0]
        
        for key in client_models[0].keys():
            # Standard FedAvg
            avg_update = sum(w * model[key] 
                           for w, model in zip(normalized_weights, client_models))
            
            # Proximal term
            prox_term = self.mu * (avg_update - self.global_model[key])
            aggregated_model[key] = avg_update + prox_term
            
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
                 model_save_path: str = "federated_models"):
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
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.model.named_parameters():
                param.data.copy_(aggregated_params[name])
        
        self.round += 1
        self._save_model()
        
    def _save_model(self):
        """Save the current global model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_save_path / f"global_model_round_{self.round}_{timestamp}.pt"
        torch.save(self.global_model.model.state_dict(), save_path)
        logger.info(f"Saved global model to {save_path}")
        
    def distribute_model(self, clients: List[FederatedClient]):
        """Distribute global model to all clients"""
        global_params = {name: param.data.clone() 
                        for name, param in self.global_model.model.named_parameters()}
        for client in clients:
            client.update_model_params(global_params)
            
            
            
class EnhancedFederatedServer(FederatedServer):
    """Enhanced server with metrics tracking and dynamic weighting"""
    def __init__(self, 
                 aggregator: FederatedAggregator,
                 model_save_path: str = "federated_models",
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