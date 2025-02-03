import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import numpy as np
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and compares performance of different aggregation strategies"""
    def __init__(self, save_dir: str = "federation_metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = defaultdict(list)
        
    def add_metric(self, strategy: str, round_num: int, metrics_dict: Dict):
        """Add metrics for a specific strategy and round"""
        self.metrics[strategy].append({
            'round': round_num,
            **metrics_dict
        })
        
    def save_metrics(self):
        """Save metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"metrics_{timestamp}.json"
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def compare_strategies(self) -> Dict:
        """Compare different aggregation strategies"""
        comparison = {}
        for strategy, rounds in self.metrics.items():
            comparison[strategy] = {
                'final_map': rounds[-1].get('mAP', 0),
                'convergence_rate': self._calculate_convergence_rate(rounds),
                'stability': self._calculate_stability(rounds)
            }
        return comparison
    
    def _calculate_convergence_rate(self, rounds: List[Dict]) -> float:
        """Calculate how quickly the model converges"""
        maps = [r.get('mAP', 0) for r in rounds]
        if len(maps) < 2:
            return 0.0
        return np.mean(np.diff(maps))
    
    def _calculate_stability(self, rounds: List[Dict]) -> float:
        """Calculate training stability"""
        maps = [r.get('mAP', 0) for r in rounds]
        return np.std(maps) if maps else 0.0