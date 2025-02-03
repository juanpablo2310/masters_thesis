from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import logging
from typing import Dict, Optional
from privacy import PrivacyMechanism

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    """Represents a client in the federated learning system"""
    def __init__(self, client_id: str , config_path: str):
        self.client_id = client_id
        self.config_path = Path(config_path)
        self.model = None
        self.load_config()
        
    def load_config(self):
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
 
    @property
    def data_path(self):
        return Path(self.config['train'])
        
            
    @data_path.setter
    def data_path(self,path : Optional[Path] = None):
        return path
    
    def initialize_model(self, weights_path: str = None):
        """Initialize or update local YOLOv8n model"""
        self.model = YOLO('yolov8n.pt' if weights_path is None else weights_path)
        
    def train(self, epochs: int = 1):
        """Train the local model"""
        try:
            results = self.model.train(
                data=self.config_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            return True
        except Exception as e:
            logger.error(f"Training failed for client {self.client_id}: {str(e)}")
            return False
            
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Extract model parameters"""
        return {name: param.data.clone() for name, param in self.model.model.named_parameters()}
        
    def update_model_params(self, new_params: Dict[str, torch.Tensor]):
        """Update local model parameters"""
        try :
            with torch.no_grad():
                for name, param in self.model.model.named_parameters():
                    param.data.copy_(new_params[name])
        
        except AttributeError:
            self.initialize_model()
                    
                
class EnhancedFederatedClient(FederatedClient):
    """Enhanced client with privacy and performance tracking"""
    def __init__(self, client_id: str,  config_path: str, 
                 privacy_mechanism: PrivacyMechanism = None):
        super().__init__(client_id, config_path)
        self.privacy_mechanism = privacy_mechanism
        self.metrics_history = []
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate local model performance"""
        try:
            results = self.model.val()
            metrics = {
                'mAP': results.box.map,
                'precision': results.box.p,
                'recall': results.box.r
            }
            self.metrics_history.append(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed for client {self.client_id}: {str(e)}")
            return {'mAP': 0.0, 'precision': 0.0, 'recall': 0.0}
            
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get privacy-preserved model parameters"""
        params = super().get_model_params()
        if self.privacy_mechanism:
            params = self.privacy_mechanism.apply(params)
        return params