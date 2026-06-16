from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import logging
from typing import Dict, List, Optional
from privacy import PrivacyMechanism
from class_mask import SharedClassSpace, GradientMasker, make_label_remap_callback

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
                project='/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/runs',
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
        """Update local model parameters, safely skipping mismatched tensors"""
        if self.model is None:
            self.initialize_model()
        with torch.no_grad():
            for name, param in self.model.model.named_parameters():
                if name not in new_params:
                    continue
                new_param = new_params[name]
                if param.data.shape != new_param.shape:
                    logger.warning(
                        f"Client {self.client_id}: shape mismatch on '{name}' "
                        f"({tuple(param.data.shape)} vs {tuple(new_param.shape)}), skipping"
                    )
                    continue
                param.data.copy_(new_param)
                    
                
class EnhancedFederatedClient(FederatedClient):
    """Enhanced client with privacy and performance tracking"""
    def __init__(self, client_id: str,  config_path: str,
                 privacy_mechanism: PrivacyMechanism = None):
        super().__init__(client_id, config_path)
        self.privacy_mechanism = privacy_mechanism
        self.metrics_history: List[Dict] = []
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate local model performance"""
        try:
            results = self.model.val(
                project='/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/runs', device='cuda' if torch.cuda.is_available() else 'cpu'
                )
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


class SharedEmbeddingClient(EnhancedFederatedClient):
    """
    Client that trains on a shared global class space using gradient masking.

    The model always has `shared_class_space.total_classes` outputs (e.g. 17).
    During training, a GradientMasker zeroes out gradients for class indices
    this client does not know, and a label-remap callback shifts local (0-based)
    labels to their global position.

    In inference, callers should index the model output with `known_class_indices`
    to get only this institution's predictions.
    """

    def __init__(
        self,
        client_id: str,
        config_path: str,
        shared_class_space: SharedClassSpace,
        privacy_mechanism: PrivacyMechanism = None,
    ):
        super().__init__(client_id, config_path, privacy_mechanism)
        self.shared_class_space = shared_class_space
        self.known_class_indices: List[int] = shared_class_space.get_known_indices(client_id)
        self.class_offset: int = shared_class_space.get_class_offset(client_id)
        self.unknown_class_indices: List[int] = shared_class_space.get_unknown_indices(client_id)
        self._gradient_masker: Optional[GradientMasker] = None

    def initialize_model(self, weights_path: str = None):
        """Initialize model with the global number of classes."""
        base = 'yolov8n.pt' if weights_path is None else weights_path
        self.model = YOLO(base)

        # Override nc so the detection head has total_classes outputs
        total_nc = self.shared_class_space.total_classes
        if hasattr(self.model, 'overrides'):
            self.model.overrides['nc'] = total_nc

        logger.info(
            f"Client {self.client_id}: model initialised with nc={total_nc}, "
            f"known classes={self.known_class_indices}, offset={self.class_offset}"
        )

    def _attach_mask_and_remap(self):
        """Register gradient masker and label remap callback on the current model."""
        if self._gradient_masker is not None:
            self._gradient_masker.remove()

        self._gradient_masker = GradientMasker(self.model, self.unknown_class_indices)

        # Remove any previously registered remap callback before adding a new one
        remap_cb = make_label_remap_callback(self.class_offset)
        self.model.add_callback("on_train_batch_start", remap_cb)

    def train(self, epochs: int = 1) -> bool:
        """Train with gradient masking and label remapping active."""
        try:
            self._attach_mask_and_remap()
            results = self.model.train(
                data=self.config_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                nc=self.shared_class_space.total_classes,
                project='/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/runs',
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            return True
        except Exception as e:
            logger.error(f"SharedEmbeddingClient {self.client_id} training failed: {e}")
            return False
        finally:
            if self._gradient_masker is not None:
                self._gradient_masker.remove()
                self._gradient_masker = None

    def predict_for_client(self, source, **kwargs):
        """
        Run inference and return only this client's class predictions.

        The returned Results objects have their `boxes.cls` values remapped
        back to local (0-based) indices for this institution.
        """
        results = self.model.predict(source, **kwargs)
        known = set(self.known_class_indices)
        filtered = []
        for r in results:
            if r.boxes is None:
                filtered.append(r)
                continue
            mask = torch.tensor(
                [int(c.item()) in known for c in r.boxes.cls],
                dtype=torch.bool,
                device=r.boxes.cls.device,
            )
            r.boxes = r.boxes[mask]
            # remap global → local indices
            r.boxes.cls = r.boxes.cls - self.class_offset
            filtered.append(r)
        return filtered
