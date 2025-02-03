from typing import List,Tuple,Dict
from pathlib import Path
from sklearn.model_selection import KFold
from client import EnhancedFederatedClient
import numpy as np

class CrossValidator:
    """Implements cross-validation for federated learning"""
    def __init__(self, n_splits: int = 5, shuffle: bool = True):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle)
        self.results = []
        
    def split_data(self, client: EnhancedFederatedClient) -> List[Tuple[List[str], List[str]]]:
        """Split client data for cross-validation"""
        # Get list of image paths
        data_path = Path(client.data_path)
        image_paths = list(data_path.glob('**/*.jpg')) + list(data_path.glob('**/*.png'))
        image_paths = [str(p) for p in image_paths]
        
        # Generate splits
        return list(self.kf.split(image_paths))
    
    def validate(self, client: EnhancedFederatedClient, 
                fold_idx: int, train_idx: List[int], val_idx: List[int]) -> Dict:
        """Validate on a specific fold"""
        # Create temporary data directories
        temp_train_dir = client.data_path.parent / f"temp_train_fold_{fold_idx}"
        temp_val_dir = client.data_path.parent / f"temp_val_fold_{fold_idx}"
        
        try:
            # Set up temporary directories
            for d in [temp_train_dir, temp_val_dir]:
                d.mkdir(exist_ok=True)
                
            # Split data into temporary directories
            self._setup_fold_data(client, train_idx, val_idx, 
                                temp_train_dir, temp_val_dir)
            
            # Train and evaluate on this fold
            client.data_path = temp_train_dir
            client.train(epochs=1)
            
            # Evaluate on validation set
            client.data_path = temp_val_dir
            metrics = client.evaluate()
            
            return metrics
            
        finally:
            # Cleanup
            import shutil
            for d in [temp_train_dir, temp_val_dir]:
                if d.exists():
                    shutil.rmtree(d)
    
    def _setup_fold_data(self, client: EnhancedFederatedClient,
                        train_idx: List[int], val_idx: List[int],
                        temp_train_dir: Path, temp_val_dir: Path):
        """Set up data for a specific fold"""
        # Implementation depends on your data structure
        pass

class EarlyStoppingCallback:
    """Implements early stopping based on convergence"""
    def __init__(self, 
                 patience: int = 3,
                 min_delta: float = 1e-4,
                 window_size: int = 5):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.best_score = None
        self.counter = 0
        self.history = []
        
    def __call__(self, current_score: float) -> bool:
        """Returns True if training should stop"""
        self.history.append(current_score)
        
        # Wait for enough history
        if len(self.history) < self.window_size:
            return False
            
        # Calculate moving average
        avg_score = np.mean(self.history[-self.window_size:])
        
        if self.best_score is None:
            self.best_score = avg_score
            return False
            
        # Check if improvement is significant
        if avg_score > self.best_score + self.min_delta:
            self.best_score = avg_score
            self.counter = 0
        else:
            self.counter += 1
            
        # Check if we should stop
        return self.counter >= self.patience
    
    def reset(self):
        """Reset the early stopping state"""
        self.best_score = None
        self.counter = 0
        self.history = []
