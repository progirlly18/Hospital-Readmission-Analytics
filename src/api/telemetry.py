"""
Phase 7: in-process operational metrics for the prediction API (/metrics).

Only aggregates are kept, in memory, since the process started: request
counts by endpoint and status code, latency of /predict, model availability,
and the distribution of returned probabilities in fixed bins. No request
body, feature value or individual prediction is stored, so the endpoint
reveals nothing about any one patient. Counters reset when the process
restarts; a deployed service would export them to a metrics system instead.
"""

import threading
from collections import Counter, deque

import numpy as np

LATENCY_WINDOW = 1000
KNOWN_PATHS = ("/health", "/model-info", "/predict", "/metrics")
# Fixed probability bins, finer where this model's predictions fall.
PROBABILITY_EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0)


class Telemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = Counter()
        self.latencies_ms = deque(maxlen=LATENCY_WINDOW)
        self.predictions = 0
        self.high_risk = 0
        self.probability_counts = [0] * (len(PROBABILITY_EDGES) - 1)

    def record_request(self, path, status_code, latency_ms):
        # Any other path is counted as "other" so arbitrary URLs cannot grow memory.
        path = path if path in KNOWN_PATHS else "other"
        with self._lock:
            self.requests[(path, int(status_code))] += 1
            if path == "/predict":
                self.latencies_ms.append(float(latency_ms))

    def record_prediction(self, probability, high_risk):
        index = int(np.searchsorted(PROBABILITY_EDGES[1:-1], probability, side="right"))
        with self._lock:
            self.predictions += 1
            self.high_risk += int(high_risk)
            self.probability_counts[index] += 1

    def snapshot(self, model_loaded, model_error):
        with self._lock:
            by_endpoint = {}
            for (path, code), count in sorted(self.requests.items()):
                by_endpoint.setdefault(path, {})[str(code)] = count
            predict = by_endpoint.get("/predict", {})
            total = sum(predict.values())
            latencies = np.array(self.latencies_ms)
            return {
                "model_loaded": model_loaded,
                "model_error": model_error,
                "requests": by_endpoint,
                "predict": {
                    "requests": total,
                    "succeeded": predict.get("200", 0),
                    "rejected_invalid": predict.get("422", 0),
                    "model_unavailable": predict.get("503", 0),
                    "server_errors": sum(count for code, count in predict.items()
                                         if code.startswith("5") and code != "503"),
                    "failure_rate": (round(1 - predict.get("200", 0) / total, 4)
                                     if total else None),
                },
                "latency_ms": {
                    "window": len(latencies),
                    "p50": round(float(np.percentile(latencies, 50)), 2) if len(latencies) else None,
                    "p95": round(float(np.percentile(latencies, 95)), 2) if len(latencies) else None,
                    "max": round(float(latencies.max()), 2) if len(latencies) else None,
                },
                "predictions": {
                    "count": self.predictions,
                    "high_risk_rate": (round(self.high_risk / self.predictions, 4)
                                       if self.predictions else None),
                    "probability_bins": list(PROBABILITY_EDGES),
                    "probability_counts": list(self.probability_counts),
                },
            }
