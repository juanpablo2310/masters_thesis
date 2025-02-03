import sys
import os
# print(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))


from client import EnhancedFederatedClient
from server import FedAvg,EnhancedFederatedServer,DynamicWeighting,FedAdagrad,FedMedian,FedTrimmedMean,FedProx
from Trainer import EnhancedFederatedTrainer
from metrics import MetricsTracker
from VisualizationTools import VisualizationTools
from AnaliticTools import CrossValidator,EarlyStoppingCallback

from scripts.utils.paths import get_project_configs

DataPathClient1 = get_project_configs(f'yaml/config_melu.yaml') #os.path.join(get_project_configs() ,'yaml', f'{args.conf_file}.yaml')
DataPathClient2 = get_project_configs(f'yaml/config_un.yaml')



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
    
    # Create enhanced clients with privacy
    clients = [
        EnhancedFederatedClient("client1", DataPathClient1),
        EnhancedFederatedClient("client2", DataPathClient2)
    ]
    
    # Initialize metrics and dynamic weighting
    metrics_tracker = MetricsTracker()
    dynamic_weighting = DynamicWeighting(num_clients=2)
    
    # Try different strategies
    strategies = [
        FedAvg(),
        FedMedian(),
        FedTrimmedMean(trim_ratio=0.1),
        FedAdagrad(learning_rate=0.01),
        FedProx(mu=0.01)
    ]
    
    results = {}
    for strategy in strategies:
        # Initialize server
        server = EnhancedFederatedServer(
            aggregator=strategy,
            metrics_tracker=metrics_tracker,
            dynamic_weighting=dynamic_weighting
        )
        
        # Create and run enhanced trainer
        trainer = EnhancedFederatedTrainer(
            server=server,
            clients=clients,
            rounds=10,
            epochs_per_round=1,
            cross_validator=cross_validator,
            early_stopping=early_stopping,
            visualization_tools=viz_tools
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
