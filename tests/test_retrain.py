"""
FastAPI Retrain Endpoint Smoke Test

This script triggers the /retrain endpoint and prints
clear, structured output aligned with model_pipeline.py style.
"""

import json
from datetime import datetime

import requests

BASE_URL = "http://127.0.0.1:8000"

# ═══════════════════════════════════════════════════════════════════════
# Formatting helpers (match model_pipeline.py style)
# ═══════════════════════════════════════════════════════════════════════

def box_title(text: str) -> None:
    line = "╔" + "═" * 70 + "╗"
    print("\n" + line)
    print(f"║  {text:<66} ║")
    print("╚" + "═" * 70 + "╝")


def section(title: str, emoji: str = "🔄") -> None:
    print("\n" + "-" * 72)
    print(f"{emoji}  {title}")
    print("-" * 72)


def kv(label: str, value) -> None:
    print(f"   ├─ {label}: {value}")


def ok(text: str) -> None:
    print(f"✅ {text}")


def warn(text: str) -> None:
    print(f"⚠️  {text}")


def err(text: str) -> None:
    print(f"❌ {text}")


# ═══════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════

def test_retrain() -> bool:
    section("Testing /retrain Endpoint", "🔧")
    params = {
        "hyperparameter_tuning": False,
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "test_size": 0.2,
    }

    print("Parameters:")
    print(json.dumps(params, indent=2))
    warn("This may take a few minutes...")

    try:
        r = requests.post(f"{BASE_URL}/retrain", json=params, timeout=300)
    except requests.exceptions.Timeout:
        err("Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        err("Could not connect to API server. Start it with 'make api'.")
        return False

    kv("Status Code", r.status_code)
    if r.status_code != 200:
        err(f"Retrain failed: {r.text}")
        return False

    data = r.json()
    ok(data.get("message", "Retrain completed"))
    kv("Timestamp", data.get("training_timestamp"))

    metrics = data.get("training_metrics", {})
    section("Training Metrics", "📊")
    print("\n📈 Training Set:")
    kv("R²", f"{metrics.get('train_r2', 0):.4f}")
    kv("RMSE", f"{metrics.get('train_rmse', 0):.4f}")
    kv("MAE", f"{metrics.get('train_mae', 0):.4f}")

    print("\n📉 Test Set:")
    kv("R²", f"{metrics.get('test_r2', 0):.4f}")
    kv("RMSE", f"{metrics.get('test_rmse', 0):.4f}")
    kv("MAE", f"{metrics.get('test_mae', 0):.4f}")
    kv("MAPE", f"{metrics.get('test_mape', 0):.2f}%")

    diff = metrics.get("train_r2", 0) - metrics.get("test_r2", 0)
    section("Overfitting Check", "🧪")
    kv("R² Difference", f"{diff:.4f}")
    if diff > 0.1:
        warn("Model may be overfitting")
    else:
        ok("Model generalizes well")

    section("Hyperparameters Used", "⚙️")
    for k, v in data.get("hyperparameters", {}).items():
        kv(k, v)

    return True


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    box_title("FastAPI Retrain Smoke Test")
    kv("API Base URL", BASE_URL)
    kv("Start Time", datetime.now().isoformat())

    success = test_retrain()
    print("\n" + ("═" * 72))
    if success:
        print("🎉 Retraining test passed! You can now run batch predictions.")
    else:
        print("⚠️  Retraining test failed. Check logs above.")
    print("" + ("═" * 72))


if __name__ == "__main__":
    main()
