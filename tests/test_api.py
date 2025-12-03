"""
FastAPI Application Smoke Tests

This script verifies the core FastAPI endpoints and prints
well-structured, friendly, and informative output matching
the style used in the ML pipeline (model_pipeline.py).
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


def section(title: str, emoji: str = "🔎") -> None:
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
# Endpoint tests
# ═══════════════════════════════════════════════════════════════════════

def test_root() -> bool:
    section("Testing Root Endpoint", "🏁")
    r = requests.get(f"{BASE_URL}/")
    kv("Status Code", r.status_code)
    print(json.dumps(r.json(), indent=2))
    return r.status_code == 200


def test_health() -> bool:
    section("Testing Health Check", "🩺")
    r = requests.get(f"{BASE_URL}/health")
    kv("Status Code", r.status_code)
    data = r.json()
    kv("Status", data.get("status"))
    kv("Model Loaded", data.get("model_loaded"))
    kv("Timestamp", data.get("timestamp"))
    return r.status_code == 200 and data.get("model_loaded") is True


def test_model_info() -> bool:
    section("Testing Model Info", "📦")
    r = requests.get(f"{BASE_URL}/model-info")
    kv("Status Code", r.status_code)
    data = r.json()
    kv("Model Path", data.get("model_path"))
    kv("Scaler Path", data.get("scaler_path"))
    kv("Model Exists", data.get("model_exists"))
    kv("Scaler Exists", data.get("scaler_exists"))
    return r.status_code == 200 and data.get("model_exists") and data.get("scaler_exists")


def test_predict() -> bool:
    section("Testing /predict", "🔮")
    payload = {
        "date": "2015-01-01",
        "quarter": "Quarter1",
        "department": "sweing",
        "day": "Thursday",
        "team": 8.0,
        "targeted_productivity": 0.8,
        "smv": 26.16,
        "wip": 1108.0,
        "over_time": 7080.0,
        "incentive": 98.0,
        "idle_time": 0.0,
        "idle_men": 0.0,
        "no_of_style_change": 0.0,
        "no_of_workers": 59.0,
    }
    print("Input:")
    print(json.dumps(payload, indent=2))
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    kv("Status Code", r.status_code)
    if r.status_code != 200:
        err(f"Prediction failed: {r.text}")
        return False
    data = r.json()
    ok(f"Predicted Productivity: {data['predicted_productivity']:.4f}")
    kv("Timestamp", data.get("prediction_timestamp"))
    return True


def test_predict_batch() -> bool:
    section("Testing /predict-batch", "📚")
    payload = {
        "samples": [
            {
                "date": "2015-01-01",
                "quarter": "Quarter1",
                "department": "sweing",
                "day": "Thursday",
                "team": 8.0,
                "targeted_productivity": 0.8,
                "smv": 26.16,
                "wip": 1108.0,
                "over_time": 7080.0,
                "incentive": 98.0,
                "idle_time": 0.0,
                "idle_men": 0.0,
                "no_of_style_change": 0.0,
                "no_of_workers": 59.0,
            },
            {
                "date": "2015-01-01",
                "quarter": "Quarter1",
                "department": "finishing",
                "day": "Thursday",
                "team": 1.0,
                "targeted_productivity": 0.75,
                "smv": 3.94,
                "wip": 500.0,
                "over_time": 960.0,
                "incentive": 0.0,
                "idle_time": 0.0,
                "idle_men": 0.0,
                "no_of_style_change": 0.0,
                "no_of_workers": 8.0,
            },
            {
                "date": "2015-01-01",
                "quarter": "Quarter1",
                "department": "sweing",
                "day": "Thursday",
                "team": 11.0,
                "targeted_productivity": 0.8,
                "smv": 11.41,
                "wip": 968.0,
                "over_time": 3660.0,
                "incentive": 50.0,
                "idle_time": 0.0,
                "idle_men": 0.0,
                "no_of_style_change": 0.0,
                "no_of_workers": 30.5,
            },
        ]
    }
    kv("Samples", len(payload["samples"]))
    r = requests.post(f"{BASE_URL}/predict-batch", json=payload)
    kv("Status Code", r.status_code)
    if r.status_code != 200:
        err(f"Batch prediction failed: {r.text}")
        return False
    data = r.json()
    ok(f"Predictions Count: {data['count']}")
    print("\n📊 Statistics:")
    for k, v in data["statistics"].items():
        kv(k, f"{v:.4f}")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    box_title("FastAPI Application Smoke Tests")
    kv("API Base URL", BASE_URL)
    kv("Start Time", datetime.now().isoformat())

    results = {
        "Root": test_root(),
        "Health": test_health(),
        "Model Info": test_model_info(),
        "Predict": test_predict(),
        "Batch Predict": test_predict_batch(),
    }

    section("Test Summary", "📋")
    all_ok = True
    for name, res in results.items():
        status = "✅ PASSED" if res else "❌ FAILED"
        print(f"  {name}: {status}")
        all_ok = all_ok and res

    print("\n" + ("═" * 72))
    if all_ok:
        print("🎉 All smoke tests passed!")
    else:
        print("⚠️  Some smoke tests failed")
    print("" + ("═" * 72))


if __name__ == "__main__":
    main()
