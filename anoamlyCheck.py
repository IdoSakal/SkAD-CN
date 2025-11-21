import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score



class CitationAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_estimators=100
        )

    def detect_anomalies(self, predicted_dynamics, actual_dynamics):
        """
        Detect anomalies by comparing predicted vs actual citation dynamics

        Args:
            predicted_dynamics: Model predictions (batch_size, horizon, node_num, feature_dim)
            actual_dynamics: Ground truth (batch_size, horizon, node_num, feature_dim)

        Returns:
            anomaly_scores: Deviation scores for each node
            anomaly_labels: Binary labels (1 = anomaly, -1 = normal)
        """
        # Calculate deviation scores for each node
        deviation_scores = self._calculate_deviation_scores(predicted_dynamics, actual_dynamics)

        # Fit isolation forest and predict anomalies
        anomaly_labels = self.isolation_forest.fit_predict(deviation_scores.reshape(-1, 1))

        return deviation_scores, anomaly_labels

    def _calculate_deviation_scores(self, predicted, actual):
        """Calculate deviation scores between predicted and actual dynamics"""
        # Mean Absolute Error for each node across time and features
        mae_per_node = np.mean(np.abs(predicted - actual), axis=(0, 1, 3))  # Average across batch, time, features
        return mae_per_node

    def evaluate_detection(self, predicted_labels, true_labels):
        """Evaluate anomaly detection performance"""
        # Convert isolation forest labels (-1, 1) to (0, 1)
        predicted_binary = (predicted_labels == 1).astype(int)
        true_binary = true_labels.astype(int)

        precision = precision_score(true_binary, predicted_binary)
        recall = recall_score(true_binary, predicted_binary)

        return precision, recall

    def inject_synthetic_anomalies(self, graph, ratio=0.1):
        """
        Inject synthetic anomalies for testing

        Args:
            graph: NetworkX graph
            ratio: Proportion of nodes to make anomalous

        Returns:
            anomalous_nodes: List of node IDs that are anomalous
            ground_truth: Binary array (1 = anomaly, 0 = normal)
        """
        node_list = list(graph.nodes())
        n_anomalies = int(len(node_list) * ratio)

        # Randomly select nodes to be anomalous
        anomalous_nodes = np.random.choice(node_list, n_anomalies, replace=False)

        # Create ground truth labels
        ground_truth = np.zeros(len(node_list))
        for i, node in enumerate(node_list):
            if node in anomalous_nodes:
                ground_truth[i] = 1

        return anomalous_nodes, ground_truth


conf = OmegaConf.load('config.yaml')
print("="*50)
print("STEP 5: Testing Model")

# Load test results
results = np.load("result.npz")
predicted_dynamics = results['predict']  # Shape: (n_samples, horizon, node_num, feature_dim)
actual_dynamics = results['ground_truth']

print(f"Prediction shape: {predicted_dynamics.shape}")
print(f"Ground truth shape: {actual_dynamics.shape}")



print("="*50)
print("STEP 6: Anomaly Detection")
print("="*50)

# Initialize anomaly detector
detector = CitationAnomalyDetector(contamination=0.2)  # Expect 10% anomalies

# Detect anomalies
deviation_scores, anomaly_labels = detector.detect_anomalies(predicted_dynamics, actual_dynamics)

# Count anomalies
n_anomalies = np.sum(anomaly_labels == 1)
n_normal = np.sum(anomaly_labels == -1)

print(f"Total nodes analyzed: {len(anomaly_labels)}")
print(f"Anomalous nodes detected: {n_anomalies}")
print(f"Normal nodes: {n_normal}")
print(f"Anomaly rate: {n_anomalies/len(anomaly_labels)*100:.2f}%")

# Get top anomalous nodes
anomalous_indices = np.where(anomaly_labels == 1)[0]
top_anomalies = anomalous_indices[np.argsort(deviation_scores[anomalous_indices])[-10:]]  # Top 10

print("\nTop 10 Most Anomalous Nodes:")
print("Node ID | Deviation Score")
print("-" * 25)
for idx in reversed(top_anomalies):
    print(f"{idx:7d} | {deviation_scores[idx]:13.6f}")
