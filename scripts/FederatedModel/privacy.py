from abc import ABC, abstractmethod
from typing import Dict,List
import torch
import numpy as np


class PrivacyMechanism(ABC):
    """Abstract base class for privacy mechanisms"""
    @abstractmethod  
    def apply(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

class DifferentialPrivacy(PrivacyMechanism):
    """Implements differential privacy using Gaussian mechanism"""
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        
    def apply(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        private_params = {}
        for name, param in params.items():
            sensitivity = torch.norm(param) / len(param.view(-1))
            sigma = np.sqrt(2 * np.log(1.25/self.delta)) * sensitivity / self.epsilon
            noise = torch.normal(0, sigma, param.shape, device=param.device)
            private_params[name] = param + noise
        return private_params

class SecureAggregation(PrivacyMechanism):
    """Simulates secure aggregation using mask-based approach"""
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.masks = self._generate_masks()
        
    def _generate_masks(self) -> List[Dict[str, torch.Tensor]]:
        """Generate random masks that sum to zero"""
        return [{}] * self.num_clients  # Placeholder for actual implementation
        
    def apply(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Simplified secure aggregation simulation
        return params