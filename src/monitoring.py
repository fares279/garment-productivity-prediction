"""
MLOps Monitoring Module

This module provides monitoring capabilities for the ML pipeline:
- MLflow metrics logging
- Elasticsearch integration
- System resource monitoring
- Model drift detection
"""

import os
import json
import time
import psutil
import socket
from datetime import datetime
from typing import Dict, Any, Optional, List
from elasticsearch import Elasticsearch, helpers
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ═════════════════════════════════════════════════════════════════════════════════
# ELASTICSEARCH CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════════

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))


class ElasticsearchLogger:
    """
    Logger for sending MLflow metrics and logs to Elasticsearch
    """

    def __init__(
        self,
        host: str = ELASTICSEARCH_HOST,
        port: int = ELASTICSEARCH_PORT,
        index_prefix: str = "mlflow",
    ):
        """
        Initialize Elasticsearch connection

        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_prefix: Prefix for Elasticsearch indices
        """
        self.host = host
        self.port = port
        self.index_prefix = index_prefix
        self.es = None
        self._connect()

    def _connect(self):
        """Connect to Elasticsearch"""
        try:
            # Elasticsearch 9.x connection syntax
            self.es = Elasticsearch(
                f"http://{self.host}:{self.port}",
                verify_certs=False,
                request_timeout=30,
            )

            # Test connection
            if self.es.ping():
                print(f"✅ Connected to Elasticsearch at {self.host}:{self.port}")
            else:
                print(f"⚠️  Could not ping Elasticsearch at {self.host}:{self.port}")
                self.es = None
        except Exception as e:
            print(f"❌ Failed to connect to Elasticsearch: {str(e)}")
            self.es = None

    def is_connected(self) -> bool:
        """Check if connected to Elasticsearch"""
        return self.es is not None and self.es.ping()

    def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: int = 0, timestamp: Optional[str] = None
    ):
        """
        Log metrics to Elasticsearch

        Args:
            run_id: MLflow run ID
            metrics: Dictionary of metric names and values
            step: Training step/epoch
            timestamp: ISO timestamp (generated if not provided)
        """
        if not self.is_connected():
            print("⚠️  Elasticsearch not connected. Skipping metric logging.")
            return

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Prepare document
        doc = {
            "run_id": run_id,
            "timestamp": timestamp,
            "step": step,
            "metrics": metrics,
            "hostname": socket.gethostname(),
        }

        # Index to Elasticsearch
        try:
            index_name = f"{self.index_prefix}-metrics"
            response = self.es.index(index=index_name, document=doc)
            print(f"✅ Logged metrics to Elasticsearch (index: {index_name})")
        except Exception as e:
            print(f"❌ Failed to log metrics to Elasticsearch: {str(e)}")

    def log_params(self, run_id: str, params: Dict[str, Any]):
        """
        Log parameters to Elasticsearch

        Args:
            run_id: MLflow run ID
            params: Dictionary of parameter names and values
        """
        if not self.is_connected():
            print("⚠️  Elasticsearch not connected. Skipping parameter logging.")
            return

        doc = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "params": params,
            "hostname": socket.gethostname(),
        }

        try:
            index_name = f"{self.index_prefix}-params"
            self.es.index(index=index_name, document=doc)
            print(f"✅ Logged parameters to Elasticsearch (index: {index_name})")
        except Exception as e:
            print(f"❌ Failed to log parameters to Elasticsearch: {str(e)}")

    def log_model_info(self, run_id: str, model_info: Dict[str, Any]):
        """
        Log model information to Elasticsearch

        Args:
            run_id: MLflow run ID
            model_info: Model metadata
        """
        if not self.is_connected():
            print("⚠️  Elasticsearch not connected. Skipping model info logging.")
            return

        doc = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": model_info,
            "hostname": socket.gethostname(),
        }

        try:
            index_name = f"{self.index_prefix}-models"
            self.es.index(index=index_name, document=doc)
            print(f"✅ Logged model info to Elasticsearch (index: {index_name})")
        except Exception as e:
            print(f"❌ Failed to log model info to Elasticsearch: {str(e)}")

    def log_system_metrics(self, run_id: str):
        """
        Log system resource metrics to Elasticsearch

        Args:
            run_id: MLflow run ID
        """
        if not self.is_connected():
            return

        # Gather system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        doc = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
            },
            "hostname": socket.gethostname(),
        }

        try:
            index_name = f"{self.index_prefix}-system"
            self.es.index(index=index_name, document=doc)
        except Exception as e:
            print(f"❌ Failed to log system metrics: {str(e)}")

    def log_prediction(
        self, run_id: str, features: Dict[str, Any], prediction: float, confidence: Optional[float] = None
    ):
        """
        Log individual prediction to Elasticsearch

        Args:
            run_id: MLflow run ID
            features: Input features
            prediction: Model prediction
            confidence: Prediction confidence score
        """
        if not self.is_connected():
            return

        doc = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "hostname": socket.gethostname(),
        }

        try:
            index_name = f"{self.index_prefix}-predictions"
            self.es.index(index=index_name, document=doc)
        except Exception as e:
            print(f"❌ Failed to log prediction: {str(e)}")

    def bulk_log_predictions(self, run_id: str, predictions_data: List[Dict[str, Any]]):
        """
        Bulk log predictions to Elasticsearch

        Args:
            run_id: MLflow run ID
            predictions_data: List of prediction dictionaries
        """
        if not self.is_connected():
            return

        actions = []
        index_name = f"{self.index_prefix}-predictions"

        for pred_data in predictions_data:
            doc = {
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                **pred_data,
                "hostname": socket.gethostname(),
            }
            actions.append({"_index": index_name, "_source": doc})

        try:
            helpers.bulk(self.es, actions)
            print(f"✅ Bulk logged {len(actions)} predictions to Elasticsearch")
        except Exception as e:
            print(f"❌ Failed to bulk log predictions: {str(e)}")


# ═════════════════════════════════════════════════════════════════════════════════
# DATA DRIFT DETECTOR
# ═════════════════════════════════════════════════════════════════════════════════


class DataDriftDetector:
    """
    Detect data drift using statistical tests
    """

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector with reference data

        Args:
            reference_data: Reference/baseline data for comparison
        """
        self.reference_data = reference_data
        self.numeric_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()

    def detect_drift(self, current_data: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect drift using Kolmogorov-Smirnov test

        Args:
            current_data: Current production data
            threshold: P-value threshold for drift detection

        Returns:
            Dictionary with drift detection results
        """
        drift_results = {}

        for column in self.numeric_columns:
            if column not in current_data.columns:
                continue

            # Perform KS test
            statistic, p_value = ks_2samp(self.reference_data[column], current_data[column])

            drift_results[column] = {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < threshold,
                "reference_mean": float(self.reference_data[column].mean()),
                "current_mean": float(current_data[column].mean()),
                "reference_std": float(self.reference_data[column].std()),
                "current_std": float(current_data[column].std()),
            }

        # Calculate overall drift score
        drift_scores = [r["ks_statistic"] for r in drift_results.values()]
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0

        # Count drifted features
        drifted_features = [col for col, r in drift_results.items() if r["drift_detected"]]

        return {
            "overall_drift_score": float(overall_drift_score),
            "drifted_features": drifted_features,
            "drift_detected": len(drifted_features) > 0,
            "features_checked": len(drift_results),
            "feature_results": drift_results,
        }

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)

        Args:
            expected: Expected/reference data
            actual: Actual/current data
            buckets: Number of buckets for discretization

        Returns:
            PSI value
        """

        def psi_for_single_feature(expected, actual, buckets):
            # Create buckets
            breakpoints = np.arange(0, buckets + 1) / buckets * 100
            breakpoints = np.unique(np.percentile(expected, breakpoints))

            # Calculate percentages in each bucket
            expected_counts = np.histogram(expected, breakpoints)[0]
            actual_counts = np.histogram(actual, breakpoints)[0]

            # Avoid division by zero
            expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
            actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

            expected_pct = expected_counts / len(expected)
            actual_pct = actual_counts / len(actual)

            # Calculate PSI
            psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi = np.sum(psi_values)

            return psi

        return float(psi_for_single_feature(expected, actual, buckets))


# ═════════════════════════════════════════════════════════════════════════════════
# SYSTEM MONITOR
# ═════════════════════════════════════════════════════════════════════════════════


class SystemMonitor:
    """
    Monitor system resources (CPU, Memory, Disk)
    """

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get current system information"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "hostname": socket.gethostname(),
            "cpu": {
                "percent_avg": sum(cpu_percent) / len(cpu_percent),
                "percent_per_core": cpu_percent,
                "count": psutil.cpu_count(),
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": disk.percent,
            },
        }

    @staticmethod
    def check_docker_containers() -> List[Dict[str, Any]]:
        """Check running Docker containers"""
        try:
            import docker

            client = docker.from_env()
            containers = []

            for container in client.containers.list():
                containers.append(
                    {
                        "id": container.id[:12],
                        "name": container.name,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                    }
                )

            return containers
        except Exception as e:
            print(f"⚠️  Could not check Docker containers: {str(e)}")
            return []


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN MONITORING CLASS
# ═════════════════════════════════════════════════════════════════════════════════


class MLOpsMonitor:
    """
    Comprehensive MLOps monitoring integrating MLflow and Elasticsearch
    """

    def __init__(
        self,
        elasticsearch_host: str = ELASTICSEARCH_HOST,
        elasticsearch_port: int = ELASTICSEARCH_PORT,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLOps monitoring

        Args:
            elasticsearch_host: Elasticsearch host
            elasticsearch_port: Elasticsearch port
            mlflow_tracking_uri: MLflow tracking server URI
        """
        # Initialize Elasticsearch logger
        self.es_logger = ElasticsearchLogger(host=elasticsearch_host, port=elasticsearch_port)

        # Set MLflow tracking URI
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        # System monitor
        self.system_monitor = SystemMonitor()

        print("\n" + "═" * 80)
        print("🔍 MLOps Monitor Initialized")
        print("═" * 80)
        print(f"   ├─ Elasticsearch: {elasticsearch_host}:{elasticsearch_port}")
        print(f"   ├─ Connection: {'✅ Connected' if self.es_logger.is_connected() else '❌ Disconnected'}")
        print(f"   └─ MLflow URI: {mlflow.get_tracking_uri()}")
        print("═" * 80 + "\n")

    def start_monitored_run(self, run_name: str, experiment_name: Optional[str] = None) -> str:
        """
        Start a monitored MLflow run

        Args:
            run_name: Name for the run
            experiment_name: Experiment name (optional)

        Returns:
            Run ID
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        mlflow.start_run(run_name=run_name)
        run_id = mlflow.active_run().info.run_id

        # Log system metrics at start
        self.es_logger.log_system_metrics(run_id)

        return run_id

    def log_metrics_monitored(self, metrics: Dict[str, float], step: int = 0):
        """
        Log metrics to both MLflow and Elasticsearch

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        run_id = mlflow.active_run().info.run_id

        # Log to MLflow
        mlflow.log_metrics(metrics, step=step)

        # Log to Elasticsearch
        self.es_logger.log_metrics(run_id, metrics, step=step)

    def log_params_monitored(self, params: Dict[str, Any]):
        """
        Log parameters to both MLflow and Elasticsearch

        Args:
            params: Dictionary of parameters
        """
        run_id = mlflow.active_run().info.run_id

        # Log to MLflow
        mlflow.log_params(params)

        # Log to Elasticsearch
        self.es_logger.log_params(run_id, params)

    def end_monitored_run(self):
        """End the monitored MLflow run"""
        run_id = mlflow.active_run().info.run_id

        # Log final system metrics
        self.es_logger.log_system_metrics(run_id)

        # End MLflow run
        mlflow.end_run()

        print(f"✅ Ended monitored run: {run_id}")


# ═════════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 80)
    print("MLOps Monitoring Module")
    print("═" * 80)
    print("\nThis module provides:")
    print("  • ElasticsearchLogger - Log metrics to Elasticsearch")
    print("  • DataDriftDetector - Detect data drift")
    print("  • SystemMonitor - Monitor system resources")
    print("  • MLOpsMonitor - Comprehensive monitoring")
    print("\n" + "═" * 80)
