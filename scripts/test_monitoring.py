"""
Test Monitoring Integration

This script tests the monitoring stack integration with:
- Elasticsearch connectivity
- MLflow logging to Elasticsearch
- System metrics collection
- Data drift detection
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Import monitoring module
try:
    from monitoring import (
        ElasticsearchLogger,
        DataDriftDetector,
        SystemMonitor,
        MLOpsMonitor,
    )

    print("✅ Monitoring module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import monitoring module: {e}")
    sys.exit(1)


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_elasticsearch_connection():
    """Test Elasticsearch connection"""
    print_section("Test 1: Elasticsearch Connection")

    logger = ElasticsearchLogger()

    if logger.is_connected():
        print("✅ Successfully connected to Elasticsearch")
        print(f"   Host: {logger.host}:{logger.port}")
        return True
    else:
        print("❌ Failed to connect to Elasticsearch")
        print("   Make sure Elasticsearch is running: make monitoring-up")
        return False


def test_log_metrics():
    """Test logging metrics to Elasticsearch"""
    print_section("Test 2: Log Metrics to Elasticsearch")

    logger = ElasticsearchLogger()

    if not logger.is_connected():
        print("⚠️  Skipping test - Elasticsearch not connected")
        return False

    # Create test metrics
    test_run_id = f"test_run_{int(time.time())}"
    test_metrics = {"accuracy": 0.95, "loss": 0.05, "f1_score": 0.93}

    logger.log_metrics(test_run_id, test_metrics, step=1)

    print(f"✅ Logged metrics to Elasticsearch")
    print(f"   Run ID: {test_run_id}")
    print(f"   Metrics: {test_metrics}")

    return True


def test_log_parameters():
    """Test logging parameters to Elasticsearch"""
    print_section("Test 3: Log Parameters to Elasticsearch")

    logger = ElasticsearchLogger()

    if not logger.is_connected():
        print("⚠️  Skipping test - Elasticsearch not connected")
        return False

    test_run_id = f"test_run_{int(time.time())}"
    test_params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model_type": "RandomForest",
    }

    logger.log_params(test_run_id, test_params)

    print(f"✅ Logged parameters to Elasticsearch")
    print(f"   Run ID: {test_run_id}")
    print(f"   Params: {test_params}")

    return True


def test_system_metrics():
    """Test system metrics collection"""
    print_section("Test 4: System Metrics Collection")

    logger = ElasticsearchLogger()
    monitor = SystemMonitor()

    # Get system info
    sys_info = monitor.get_system_info()

    print("✅ System metrics collected")
    print(f"   CPU: {sys_info['cpu']['percent_avg']:.1f}%")
    print(f"   Memory: {sys_info['memory']['percent']:.1f}%")
    print(f"   Disk: {sys_info['disk']['percent']:.1f}%")

    # Log to Elasticsearch if connected
    if logger.is_connected():
        test_run_id = f"test_run_{int(time.time())}"
        logger.log_system_metrics(test_run_id)
        print(f"   ✅ Logged to Elasticsearch (Run ID: {test_run_id})")

    return True


def test_data_drift_detection():
    """Test data drift detection"""
    print_section("Test 5: Data Drift Detection")

    # Create reference data
    np.random.seed(42)
    reference_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000),
            "feature3": np.random.uniform(0, 10, 1000),
        }
    )

    # Create current data with drift
    current_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0.5, 1.2, 1000),  # Mean shift
            "feature2": np.random.normal(5, 2, 1000),  # No drift
            "feature3": np.random.uniform(2, 12, 1000),  # Range shift
        }
    )

    # Detect drift
    detector = DataDriftDetector(reference_data)
    drift_results = detector.detect_drift(current_data, threshold=0.05)

    print(f"✅ Drift detection completed")
    print(f"   Overall drift score: {drift_results['overall_drift_score']:.4f}")
    print(f"   Drifted features: {drift_results['drifted_features']}")
    print(f"   Drift detected: {drift_results['drift_detected']}")

    # Show details for each feature
    for feature, result in drift_results["feature_results"].items():
        status = "🔴 DRIFT" if result["drift_detected"] else "✅ OK"
        print(
            f"   {feature}: {status} (KS={result['ks_statistic']:.4f}, p={result['p_value']:.4f})"
        )

    return True


def test_mlops_monitor():
    """Test MLOps Monitor integration"""
    print_section("Test 6: MLOps Monitor Integration")

    try:
        # Initialize monitor
        monitor = MLOpsMonitor(mlflow_tracking_uri="http://127.0.0.1:5000")

        print("✅ MLOpsMonitor initialized successfully")

        # Test if Elasticsearch is connected
        if monitor.es_logger.is_connected():
            print("   ✅ Elasticsearch connected")
        else:
            print("   ⚠️  Elasticsearch not connected")

        return True
    except Exception as e:
        print(f"❌ MLOpsMonitor test failed: {str(e)}")
        return False


def test_docker_containers():
    """Test Docker container monitoring"""
    print_section("Test 7: Docker Container Status")

    monitor = SystemMonitor()
    containers = monitor.check_docker_containers()

    if containers:
        print(f"✅ Found {len(containers)} running Docker containers:")
        for container in containers:
            print(f"   • {container['name']} ({container['status']})")
    else:
        print("⚠️  No Docker containers found or Docker not accessible")

    return True


def main():
    """Run all tests"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🧪 Monitoring Integration Tests" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")

    print(f"\n📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    tests = [
        test_elasticsearch_connection,
        test_log_metrics,
        test_log_parameters,
        test_system_metrics,
        test_data_drift_detection,
        test_mlops_monitor,
        test_docker_containers,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append((test.__name__, False))

    # Print summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} - {test_name}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("  1. View Kibana dashboard: make kibana-open")
        print("  2. Run training with monitoring: make full-pipeline")
        print("  3. Check Elasticsearch indices: make elasticsearch-check")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
        print("\nTroubleshooting:")
        print("  1. Make sure monitoring stack is running: make monitoring-status")
        print("  2. Check logs: make monitoring-logs")
        print("  3. Restart stack: make monitoring-down && make monitoring-up")

    print("")


if __name__ == "__main__":
    main()
