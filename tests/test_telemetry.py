import numpy as np
import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import create_app  # noqa: E402
from src.api.model import EXPECTED_THRESHOLD, LoadedModel, ModelNotReady, Settings  # noqa: E402
from src.api.telemetry import PROBABILITY_EDGES, Telemetry  # noqa: E402
from src.ml import config  # noqa: E402
from tests.test_api import valid_request  # noqa: E402


class FixedModel:
    """Returns the same probability for every row; enough to test telemetry."""
    classes_ = [0, 1]

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        return np.tile([1 - self.probability, self.probability], (len(frame), 1))


def loader_for(probability):
    def load(settings):
        return LoadedModel(model=FixedModel(probability), name="test-model", version="1",
                           alias="candidate", threshold=EXPECTED_THRESHOLD,
                           model_sha256="0" * 64, serialization_format="skops",
                           feature_columns=tuple(config.FEATURE_COLUMNS))
    return load


def unavailable(settings):
    raise ModelNotReady("store unreachable")


def test_metrics_count_outcomes_latency_and_probabilities():
    with TestClient(create_app(Settings(), loader=loader_for(0.3))) as client:
        for _ in range(3):
            assert client.post("/predict", json=valid_request()).status_code == 200
        assert client.post("/predict", json=valid_request(race="x")).status_code == 422
        client.get("/health")
        metrics = client.get("/metrics").json()

    assert metrics["model_loaded"] is True and metrics["model_error"] is None
    predict = metrics["predict"]
    assert (predict["requests"], predict["succeeded"], predict["rejected_invalid"]) == (4, 3, 1)
    assert predict["failure_rate"] == 0.25
    assert metrics["requests"]["/health"] == {"200": 1}
    assert metrics["latency_ms"]["window"] == 4 and metrics["latency_ms"]["p95"] >= 0
    counts = metrics["predictions"]["probability_counts"]
    assert sum(counts) == 3 and counts[PROBABILITY_EDGES.index(0.30)] == 3
    assert metrics["predictions"]["high_risk_rate"] == 1.0


def test_metrics_report_an_unavailable_model():
    with TestClient(create_app(Settings(), loader=unavailable)) as client:
        assert client.post("/predict", json=valid_request()).status_code == 503
        metrics = client.get("/metrics").json()
    assert metrics["model_loaded"] is False and metrics["model_error"] == "store unreachable"
    assert metrics["predict"]["model_unavailable"] == 1
    assert metrics["predictions"]["count"] == 0


def leaves(node, path=()):
    """Yield (path, key or value) for every key and leaf value in a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key
            yield from leaves(value, path + (key,))
    elif isinstance(node, list):
        for value in node:
            yield from leaves(value, path)
    else:
        yield path, node


def metrics_after(*bodies):
    with TestClient(create_app(Settings(), loader=loader_for(0.05))) as client:
        for body in bodies:
            client.post("/predict", json=body)
        return client.get("/metrics").json()


def test_metrics_never_contain_request_values():
    # Compare exact keys and values rather than searching the raw text, so a
    # measured latency such as 12.77 cannot be mistaken for the input 77.
    metrics = metrics_after(
        valid_request(medical_specialty="Cardiology", gender="Male", num_medications=77),
        valid_request(encounter_id=123456789),
    )
    request_fields = set(valid_request()) | {"encounter_id"}
    for path, item in leaves(metrics):
        if isinstance(item, str):
            assert item not in request_fields
            for value in ("Cardiology", "Male", "77", "123456789", "encounter_id"):
                assert value not in item
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and path[:1] != ("latency_ms",):
            assert item not in (77, 123456789)


def test_metrics_contain_only_aggregate_fields():
    metrics = metrics_after(valid_request())
    assert set(metrics) == {"model_loaded", "model_error", "requests", "predict", "latency_ms", "predictions"}
    assert set(metrics["requests"]) <= {"/health", "/model-info", "/predict", "/metrics", "other"}
    assert set(metrics["predict"]) == {"requests", "succeeded", "rejected_invalid",
                                       "model_unavailable", "server_errors", "failure_rate"}
    assert set(metrics["latency_ms"]) == {"window", "p50", "p95", "max"}
    assert set(metrics["predictions"]) == {"count", "high_risk_rate", "probability_bins",
                                           "probability_counts"}


def test_metrics_do_not_depend_on_request_values():
    first = metrics_after(valid_request(medical_specialty="Cardiology", gender="Male", num_medications=77))
    second = metrics_after(valid_request(num_medications=12))
    first.pop("latency_ms"), second.pop("latency_ms")
    assert first == second


def test_unknown_paths_share_one_counter():
    with TestClient(create_app(Settings(), loader=loader_for(0.1))) as client:
        for index in range(5):
            client.get(f"/random-{index}")
        requests = client.get("/metrics").json()["requests"]
    assert requests["other"] == {"404": 5}
    assert not any(path.startswith("/random") for path in requests)


def test_server_errors_are_counted():
    class Broken(FixedModel):
        def predict_proba(self, frame):
            raise ValueError("boom")

    def load(settings):
        return loader_for(0.1)(settings).__class__(**{
            **loader_for(0.1)(settings).__dict__, "model": Broken(0.1)})

    app = create_app(Settings(), loader=load)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/predict", json=valid_request()).status_code == 500
        predict = client.get("/metrics").json()["predict"]
    assert predict["server_errors"] == 1 and predict["failure_rate"] == 1.0


def test_probability_bins_cover_zero_to_one():
    telemetry = Telemetry()
    for probability in (0.0, 0.049, 0.05, 0.99, 1.0):
        telemetry.record_prediction(probability, probability >= EXPECTED_THRESHOLD)
    counts = telemetry.snapshot(True, None)["predictions"]["probability_counts"]
    assert counts[0] == 2 and counts[1] == 1 and counts[-1] == 2
