from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import logging
from typing import Dict, List, Optional
import shutil
from privacy import PrivacyMechanism
from class_mask import SharedClassSpace, GradientMasker, build_global_dataset, build_shared_yolo

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
        """Full state_dict (params + BN buffers), minus the non-float BN counters.

        Including BatchNorm running stats is essential for federation: otherwise the
        shared trunk's normalization is never synchronized across clients.
        """
        return {
            k: v.detach().clone()
            for k, v in self.model.model.state_dict().items()
            if not k.endswith("num_batches_tracked")
        }

    def update_model_params(self, new_params: Dict[str, torch.Tensor]):
        """Load incoming params/buffers, keeping only matching-shape tensors."""
        if self.model is None:
            self.initialize_model()
        own = self.model.model.state_dict()
        filtered = {k: v for k, v in new_params.items() if k in own and own[k].shape == v.shape}
        skipped = len(new_params) - len(filtered)
        if skipped:
            logger.warning(f"Client {self.client_id}: skipped {skipped} mismatched tensors")
        self.model.model.load_state_dict(filtered, strict=False)
                    
                
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

    The model is trained on a generated 17-class dataset YAML (the shared space)
    so its detection head has `shared_class_space.total_classes` outputs. Labels
    are remapped to their global indices (via `build_global_dataset`), and a
    `GradientMasker` — attached on `on_train_start` so it survives ultralytics'
    model setup/device move — zeroes the gradients of the classes this client
    does not know.

    In inference, `predict_for_client` returns only this client's class
    predictions, remapped back to local (0-based) indices.
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
        self._global_tmp_root: Optional[Path] = None
        self._eval_yaml: Optional[Path] = None
        self._eval_tmp_root: Optional[Path] = None

    def initialize_model(self, weights_path: str = None):
        """Build a 17-class model so the architecture is fixed before any round.

        Weights are overwritten by the distributed global model each round; only
        the architecture (shared 17-output head) needs to be set here.
        """
        if weights_path is not None:
            self.model = YOLO(weights_path)
        else:
            self.model = build_shared_yolo(
                self.shared_class_space.total_classes,
                self.shared_class_space.get_global_names(),
            )
        logger.info(
            f"Client {self.client_id}: 17-class model ready — "
            f"known={self.known_class_indices}, offset={self.class_offset}"
        )

    @property
    def num_train_images(self) -> int:
        """Number of training images this client owns (for data-size weighting)."""
        try:
            img_dir = Path(self.config['train'])
            return sum(1 for p in img_dir.iterdir()
                       if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
        except Exception:
            return 0

    def get_eval_dataset(self) -> Path:
        """Build (and cache) the full 17-class val dataset for global evaluation."""
        if self._eval_yaml is None:
            self._eval_yaml, self._eval_tmp_root = build_global_dataset(
                self.config_path,
                self.class_offset,
                self.shared_class_space.get_global_names(),
                self.shared_class_space.total_classes,
                splits=("val",),
            )
        return self._eval_yaml

    def _snapshot_unknown_head(self):
        """Clone this client's unknown-class head rows (weights+biases) to keep frozen."""
        idx = self.unknown_class_indices
        return [p.detach()[idx].clone() for p in GradientMasker.head_param_list(self.model)]

    def _restore_unknown_head(self, snapshot):
        """Overwrite the unknown-class head rows of the current model with the snapshot.

        Applied AFTER ultralytics finishes (including its final reload from best.pt),
        so the classes this client does not own are provably unchanged — robust to
        any optimizer / EMA / callback-timing behavior inside the training loop.
        """
        if not snapshot:
            return
        idx = self.unknown_class_indices
        with torch.no_grad():
            for p, s in zip(GradientMasker.head_param_list(self.model), snapshot):
                p.data[idx] = s.to(p.dtype).to(p.device)

    def train_round(self, local_images: int = None, epochs: int = 1, seed: int = None) -> bool:
        """Run one federated round of local training from the current (global) weights.

        A fresh random subsample of ``local_images`` training images is used each
        round (varied by ``seed``) to keep local SGD steps small — the key to
        keeping the shared trunk synchronized across clients.
        """
        # Rebuild a fresh subsampled training set for this round.
        if self._global_tmp_root is not None:
            shutil.rmtree(self._global_tmp_root, ignore_errors=True)
        data_yaml, self._global_tmp_root = build_global_dataset(
            self.config_path,
            self.class_offset,
            self.shared_class_space.get_global_names(),
            self.shared_class_space.total_classes,
            splits=("train", "val"),
            max_train_images=local_images,
            seed=seed,
        )
        # Snapshot the unknown-class head rows from the just-distributed (global) model.
        unknown_snapshot = self._snapshot_unknown_head()
        try:
            self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=640,
                batch=16,
                val=False,            # global eval is done centrally by the trainer
                plots=False,
                verbose=False,
                project='/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/runs',
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            # Enforce the class mask: restore the unknown rows to their global value,
            # so this client contributes only updates to the classes it owns.
            self._restore_unknown_head(unknown_snapshot)
            return True
        except Exception as e:
            logger.error(f"SharedEmbeddingClient {self.client_id} round failed: {e}")
            return False

    # Backwards-compatible alias
    def train(self, epochs: int = 1) -> bool:
        return self.train_round(epochs=epochs)

    def cleanup(self):
        """Remove temporary dataset directories, if any."""
        for attr in ("_global_tmp_root", "_eval_tmp_root"):
            root = getattr(self, attr)
            if root is not None:
                shutil.rmtree(root, ignore_errors=True)
                setattr(self, attr, None)
        self._eval_yaml = None

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
