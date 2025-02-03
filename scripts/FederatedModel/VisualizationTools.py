import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationTools:
    """Tools for visualizing federated learning metrics"""
    def __init__(self, save_dir: str = "federation_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_convergence(self, metrics: Dict[str, List[Dict]], 
                        save_name: str = "convergence"):
        """Plot convergence of different strategies"""
        plt.figure(figsize=(12, 6))
        
        for strategy, rounds in metrics.items():
            maps = [r.get('mAP', 0) for r in rounds]
            rounds = list(range(1, len(maps) + 1))
            plt.plot(rounds, maps, marker='o', label=strategy)
            
        plt.xlabel('Round')
        plt.ylabel('mAP')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.grid(True)
        
        save_path = self.save_dir / f"{save_name}_convergence.png"
        plt.savefig(save_path)
        plt.close()
        
    def plot_performance_heatmap(self, metrics: Dict[str, List[Dict]], 
                               save_name: str = "performance"):
        """Create heatmap of strategy performance"""
        strategies = list(metrics.keys())
        metrics_names = ['mAP', 'precision', 'recall']
        
        data = []
        for strategy in strategies:
            last_round = metrics[strategy][-1]
            data.append([last_round.get(metric, 0) for metric in metrics_names])
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, 
                   xticklabels=metrics_names,
                   yticklabels=strategies,
                   annot=True,
                   cmap='YlOrRd')
        
        plt.title('Strategy Performance Comparison')
        save_path = self.save_dir / f"{save_name}_heatmap.png"
        plt.savefig(save_path)
        plt.close()
        
    def create_interactive_dashboard(self, metrics: Dict[str, List[Dict]],
                                  save_name: str = "dashboard"):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Convergence', 'Final Performance', 
                          'Client Contributions', 'Training Stability')
        )
        
        # Convergence plot
        for strategy, rounds in metrics.items():
            maps = [r.get('mAP', 0) for r in rounds]
            rounds = list(range(1, len(maps) + 1))
            fig.add_trace(
                go.Scatter(x=rounds, y=maps, name=strategy, mode='lines+markers'),
                row=1, col=1
            )
            
        # Final performance bar chart
        final_maps = [rounds[-1].get('mAP', 0) for rounds in metrics.values()]
        fig.add_trace(
            go.Bar(x=list(metrics.keys()), y=final_maps),
            row=1, col=2
        )
        
        # Save interactive dashboard
        save_path = self.save_dir / f"{save_name}_dashboard.html"
        fig.write_html(str(save_path))