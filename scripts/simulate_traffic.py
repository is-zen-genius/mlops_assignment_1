"""Send a burst of synthetic traffic to a running API instance, for populating dashboards.

Usage:
    python scripts/simulate_traffic.py --url http://localhost:8001 --requests 300 --delay 0.2
"""
import argparse
import random
import time

import requests


def random_patient() -> dict:
    return {
        "age": random.randint(29, 77),
        "sex": random.randint(0, 1),
        "cp": random.randint(1, 4),
        "trestbps": random.randint(94, 200),
        "chol": random.randint(126, 410),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(71, 202),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.random() * 5, 1),
        "slope": random.randint(1, 3),
        "ca": random.randint(0, 4),
        "thal": random.choice([3, 6, 7]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8001", help="base URL of the API")
    parser.add_argument("--requests", type=int, default=300, help="number of requests to send")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between requests")
    parser.add_argument(
        "--invalid-rate", type=float, default=0.1, help="fraction of requests that are malformed"
    )
    args = parser.parse_args()

    accepted = rejected = 0
    for _ in range(args.requests):
        body = random_patient()
        if random.random() < args.invalid_rate:
            body["age"] = 9999  # deliberately out of range -> 422, exercises the error path
        try:
            response = requests.post(f"{args.url}/predict", json=body, timeout=5)
            if response.status_code == 200:
                accepted += 1
            else:
                rejected += 1
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            rejected += 1
        time.sleep(args.delay)

    print(f"Done: {accepted} accepted (200), {rejected} rejected/failed, target={args.url}")


if __name__ == "__main__":
    main()
