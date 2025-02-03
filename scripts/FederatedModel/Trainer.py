import logging
from server import FederatedServer,EnhancedFederatedServer
from client import FederatedClient,EnhancedFederatedClient
from typing import List, Optional
import numpy as np
from AnaliticTools import CrossValidator,EarlyStoppingCallback
from VisualizationTools import VisualizationTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedTrainer:
    """Orchestrates the federated learning process"""
    def __init__(self, 
                 server: FederatedServer,
                 clients: List[FederatedClient],
                 rounds: int = 10,
                 epochs_per_round: int = 1):
        self.server = server
        self.clients = clients
        self.rounds = rounds
        self.epochs_per_round = epochs_per_round
        
    def train(self):
        """Execute federated training process"""
        logger.info("Starting federated training...")
        
        for round_num in range(self.rounds):
            logger.info(f"Starting round {round_num + 1}/{self.rounds}")
            
            # Distribute global model to clients
            self.server.distribute_model(self.clients)
            
            # Train local models
            successful_clients = []
            for client in self.clients:
                if client.train(epochs=self.epochs_per_round):
                    successful_clients.append(client)
            
            if not successful_clients:
                logger.error("No successful client training in this round")
                continue
                
            # Aggregate models
            self.server.aggregate_models(successful_clients)
            logger.info(f"Completed round {round_num + 1}")
            
        logger.info("Federated training complete")
        
class EnhancedFederatedTrainer(FederatedTrainer):
    """Enhanced trainer with cross-validation and early stopping"""
    def __init__(self,
                 server: EnhancedFederatedServer,
                 clients: List[EnhancedFederatedClient],
                 rounds: int = 10,
                 epochs_per_round: int = 1,
                 cross_validator: Optional[CrossValidator] = None,
                 early_stopping: Optional[EarlyStoppingCallback] = None,
                 visualization_tools: Optional[VisualizationTools] = None):
        super().__init__(server, clients, rounds, epochs_per_round)
        self.cross_validator = cross_validator
        self.early_stopping = early_stopping
        self.visualization_tools = visualization_tools
        
    def train(self):
        """Execute federated training with cross-validation and early stopping"""
        logger.info("Starting enhanced federated training...")
        
        if self.cross_validator:
            self._run_cross_validation()
            
        for round_num in range(self.rounds):
            logger.info(f"Starting round {round_num + 1}/{self.rounds}")
            
            # Regular training
            self.server.distribute_model(self.clients)
            successful_clients = []
            
            for client in self.clients:
                client.initialize_model()
                if client.train(epochs=self.epochs_per_round):
                    successful_clients.append(client)
                    
            if not successful_clients:
                logger.error("No successful client training in this round")
                continue
                
            # Aggregate and evaluate
            self.server.aggregate_models(successful_clients)
            current_score = self.server._evaluate_global_model()['mAP']
            
            # Visualize current state
            if self.visualization_tools:
                self.visualization_tools.plot_convergence(
                    self.server.metrics_tracker.metrics,
                    f"round_{round_num + 1}"
                )
                
            # Check early stopping
            if self.early_stopping and self.early_stopping(current_score):
                logger.info("Early stopping triggered")
                break
                
        # Final visualization
        if self.visualization_tools:
            self.visualization_tools.plot_performance_heatmap(
                self.server.metrics_tracker.metrics,
                "final"
            )
            self.visualization_tools.create_interactive_dashboard(
                self.server.metrics_tracker.metrics
            )
            
        logger.info("Enhanced federated training complete")
        
    def _run_cross_validation(self):
        """Run cross-validation before main training"""
        logger.info("Starting cross-validation...")
        
        for client in self.clients:
            cv_metrics = []
            splits = self.cross_validator.split_data(client)
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                logger.info(f"Running fold {fold_idx + 1}/{len(splits)}")
                metrics = self.cross_validator.validate(
                    client, fold_idx, train_idx, val_idx
                )
                cv_metrics.append(metrics)
                
            # Log cross-validation results
            avg_map = np.mean([m['mAP'] for m in cv_metrics])
            std_map = np.std([m['mAP'] for m in cv_metrics])
            logger.info(f"Cross-validation results for {client.client_id}:")
            logger.info(f"Average mAP: {avg_map:.4f} ± {std_map:.4f}")