import sys
import os
# print(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))


from client import EnhancedFederatedClient, SharedEmbeddingClient
from server import FedAvg,EnhancedFederatedServer,DynamicWeighting,FedAdagrad,FedMedian,FedTrimmedMean,FedProx
from Trainer import EnhancedFederatedTrainer
from metrics import MetricsTracker
from VisualizationTools import VisualizationTools
from AnaliticTools import CrossValidator,EarlyStoppingCallback
from class_mask import SharedClassSpace

import yaml
from scripts.utils.paths import get_project_configs

# client1 = UNAL (6 classes, global indices 0–5)
# client2 = Melbourne (11 classes, global indices 6–16)
DataPathUNAL = get_project_configs(f'yaml/config_un.yaml')
DataPathMelbourne = get_project_configs(f'yaml/config_melu.yaml')


def _load_class_names(config_path) -> dict:
    """Read the {index: name} class map from a YOLO dataset YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    names = cfg.get('names', {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}



def main():
    # # Create clients with different configurations
    # clients = [
    #     FederatedClient("client1", "path/to/data1", "config_melu.yaml"),
    #     FederatedClient("client2", "path/to/data2", "config_un.yaml")
    # ]
    
    # # Initialize server with FedAvg strategy
    # server = FederatedServer(aggregator=FedAvg())
    
    # # Create and run trainer
    # trainer = FederatedTrainer(server, clients, rounds=10, epochs_per_round=1)
    # trainer.train()
    
    viz_tools = VisualizationTools()
    
    # Initialize cross-validator
    cross_validator = CrossValidator(n_splits=5)
    
    # Initialize early stopping
    early_stopping = EarlyStoppingCallback(
        patience=3,
        min_delta=1e-4,
        window_size=5
    )
    
    # --- Shared embedding space ---
    # client1 = UNAL: 6 classes -> global indices 0–5
    # client2 = Melbourne: 11 classes -> global indices 6–16
    # Total model outputs: 17
    shared_space = SharedClassSpace({
        "client1": list(range(0, 6)),    # UNAL
        "client2": list(range(6, 17)),   # Melbourne
    })
    # Attach human-readable global names (UNAL at 0–5, Melbourne at 6–16)
    shared_space.assign_names({
        "client1": _load_class_names(DataPathUNAL),
        "client2": _load_class_names(DataPathMelbourne),
    })

    # Create clients with shared embedding space and class masks
    clients = [
        SharedEmbeddingClient("client1", DataPathUNAL, shared_space),
        SharedEmbeddingClient("client2", DataPathMelbourne, shared_space),
    ]
    
    # Initialize metrics
    metrics_tracker = MetricsTracker()

    # Synchronized FedAvg over the shared 17-class space.
    # (Running several strategies = several full federated trainings; start with FedAvg.)
    strategies = [
        FedAvg(),
        # FedMedian(), FedTrimmedMean(trim_ratio=0.1), FedAdagrad(0.01), FedProx(0.01),
    ]

    results = {}
    for strategy in strategies:
        # 17-class global model with a neutral pretrained backbone
        server = EnhancedFederatedServer(
            aggregator=strategy,
            num_classes=shared_space.total_classes,
            class_names=shared_space.get_global_names(),
            metrics_tracker=metrics_tracker,
        )

        # Many short rounds + small local steps keep the shared trunk synchronized
        trainer = EnhancedFederatedTrainer(
            server=server,
            clients=clients,
            rounds=20,
            epochs_per_round=1,
            local_images=300,        # random subsample per client per round
            weighting="equal",       # or "data" to weight by dataset size
            early_stopping=early_stopping,
            visualization_tools=viz_tools,
        )
        trainer.train()

        # Store results
        results[type(strategy).__name__] = metrics_tracker.compare_strategies()
    
    # Final visualization
    viz_tools.create_interactive_dashboard(metrics_tracker.metrics, "final_comparison")
    
    # Print comparison
    print("\nStrategy Comparison:")
    for strategy_name, metrics in results.items():
        print(f"\n{strategy_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")
    
if __name__ == "__main__":
    main()


