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
    """Synchronized FedAvg trainer for the shared embedding space (Option B).

    Each round: distribute the global model -> every client trains a *few* local
    SGD steps (small random subsample) with gradient masking on its class slice ->
    aggregate the full models (trunk + masked heads) -> evaluate the global model.
    Frequent synchronization keeps the shared trunk in one loss basin, so a single
    17-class model becomes good at every institution's classes.
    """
    def __init__(self,
                 server: EnhancedFederatedServer,
                 clients: List[EnhancedFederatedClient],
                 rounds: int = 20,
                 epochs_per_round: int = 1,
                 local_images: int = 300,
                 weighting: str = "equal",            # "equal" | "data"
                 early_stopping: Optional[EarlyStoppingCallback] = None,
                 visualization_tools: Optional[VisualizationTools] = None,
                 cross_validator: Optional[CrossValidator] = None):
        super().__init__(server, clients, rounds, epochs_per_round)
        self.local_images = local_images
        self.weighting = weighting
        self.early_stopping = early_stopping
        self.visualization_tools = visualization_tools
        self.cross_validator = cross_validator  # kept for API compat; not used in B

    def _client_weights(self, clients: List[EnhancedFederatedClient]) -> List[float]:
        if self.weighting == "data":
            sizes = [max(1, c.num_train_images) for c in clients]
            total = float(sum(sizes))
            return [s / total for s in sizes]
        return [1.0 / len(clients)] * len(clients)

    def _evaluate_global(self) -> float:
        """Average the global model's metrics across each client's val set.

        One disposable eval model is built per round (so fusing during val never
        touches the canonical global weights) and validated on every client.
        """
        eval_model = self.server.make_eval_model()
        per_client = []
        for client in self.clients:
            m = self.server.val_metrics(eval_model, client.get_eval_dataset())
            m["client"] = client.client_id
            per_client.append(m)
            logger.info(
                f"  [{client.client_id}] global mAP50={m['mAP50']:.4f} "
                f"mAP={m['mAP']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}"
            )
        keys = ["mAP", "mAP50", "precision", "recall"]
        avg = {k: float(np.mean([m[k] for m in per_client])) for k in keys}
        self.server.metrics_tracker.add_metric(
            type(self.server.aggregator).__name__, self.server.round, avg
        )
        return avg["mAP50"]

    def train(self):
        """Execute synchronized federated training."""
        logger.info("Starting synchronized federated training (Option B)...")

        # Fix each client's 17-class architecture once; weights come from the server.
        for client in self.clients:
            client.initialize_model()

        for round_num in range(self.rounds):
            logger.info(f"=== Round {round_num + 1}/{self.rounds} ===")

            self.server.distribute_model(self.clients)

            successful = []
            for ci, client in enumerate(self.clients):
                if client.train_round(local_images=self.local_images,
                                      epochs=self.epochs_per_round,
                                      seed=round_num * 1000 + ci):
                    successful.append(client)

            if not successful:
                logger.error("No successful client training this round")
                continue

            self.server.aggregate_models(successful, self._client_weights(successful))
            current_score = self._evaluate_global()
            logger.info(f"Round {round_num + 1}: global mAP50 = {current_score:.4f}")

            if self.visualization_tools:
                self.visualization_tools.plot_convergence(
                    self.server.metrics_tracker.metrics, f"round_{round_num + 1}"
                )

            if self.early_stopping and self.early_stopping(current_score):
                logger.info("Early stopping triggered")
                break

        for client in self.clients:
            client.cleanup()

        if self.visualization_tools:
            self.visualization_tools.create_interactive_dashboard(
                self.server.metrics_tracker.metrics
            )
        logger.info(
            f"Federated training complete. Final global model: "
            f"{getattr(self.server, 'last_save_path', '(not saved)')}"
        )