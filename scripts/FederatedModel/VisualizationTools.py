import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict
import logging
import numpy as np

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
        """Create heatmap of strategy performance.

        This method auto-normalizes common irregular inputs:
        - if a metric value is a list/array, it uses the mean
        - non-numeric values become NaN
        - builds a rectangular DataFrame (strategies x metrics)
        """
        import pandas as pd
        strategies = list(metrics.keys())
        metrics_names = ['mAP', 'precision', 'recall']

        rows = []
        for strategy in strategies:
            rounds = metrics.get(strategy, [])
            if not rounds:
                logger.warning(f"No rounds for strategy '{strategy}'; filling with NaN")
                rows.append([float('nan')] * len(metrics_names))
                continue

            last_round = rounds[-1]
            row = []
            for metric in metrics_names:
                val = last_round.get(metric, None)
                if val is None:
                    row.append(float('nan'))
                    continue

                # If value is a sequence (list/tuple/ndarray), reduce to scalar (mean)
                if isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        arr = np.array(val, dtype=float)
                        if arr.size == 0:
                            row.append(float('nan'))
                        else:
                            row.append(float(np.nanmean(arr)))
                            logger.debug(f"Reduced list metric for '{strategy}.{metric}' to mean")
                    except Exception:
                        logger.warning(f"Could not convert list metric for '{strategy}.{metric}' to numeric; using NaN")
                        row.append(float('nan'))
                    continue

                # Try to coerce to float
                try:
                    row.append(float(val))
                except Exception:
                    logger.warning(f"Non-numeric metric for '{strategy}.{metric}': {val}; using NaN")
                    row.append(float('nan'))

            rows.append(row)

        # Build DataFrame to ensure rectangular numeric input for seaborn
        try:
            df = pd.DataFrame(rows, index=strategies, columns=metrics_names, dtype=float)
        except Exception as e:
            logger.error(f"Failed to build DataFrame for heatmap: {e}")
            # fallback: coerce to numpy array of object and try to pad
            arr = np.array(rows, dtype=object)
            max_cols = max(len(r) for r in rows) if rows else 0
            mat = np.full((len(rows), max_cols), np.nan, dtype=float)
            for i, r in enumerate(rows):
                for j, v in enumerate(r):
                    try:
                        mat[i, j] = float(v)
                    except Exception:
                        mat[i, j] = np.nan
            import pandas as pd
            df = pd.DataFrame(mat, index=strategies, columns=metrics_names[:mat.shape[1]])

        plt.figure(figsize=(10, 8))
        sns.heatmap(df, 
                   xticklabels=df.columns.tolist(),
                   yticklabels=df.index.tolist(),
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