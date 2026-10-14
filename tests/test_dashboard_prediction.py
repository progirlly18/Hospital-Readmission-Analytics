import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
uvicorn = pytest.importorskip("uvicorn")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.api import client  # noqa: E402
from src.api.main import create_app  # noqa: E402
from src.api.schemas import PredictionRequest  # noqa: E402
from src.api.model import Settings  # noqa: E402
from src.ml import config  # noqa: E402
from tests.test_api import valid_request  # noqa: E402
from tests.test_telemetry import loader_for, unavailable  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "1_Readmission_Risk_Prediction.py"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def serve():
    """Run the real API app over HTTP on a free local port."""
    servers = []

    def start(loader):
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(
            create_app(Settings(), loader=loader), host="127.0.0.1", port=port,
            log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)


# ---------------------------------------------------------
# Client
# ---------------------------------------------------------

def test_client_health_and_prediction(serve):
    url = serve(loader_for(0.3))
    assert client.health(url).ok
    result = client.predict(valid_request(), url)
    assert result.ok
    assert result.data["readmission_probability"] == pytest.approx(0.3)
    assert result.data["high_risk"] is True and "disclaimer" in result.data


def test_client_reports_rejected_fields_without_values(serve):
    url = serve(loader_for(0.3))
    result = client.predict(valid_request(race="Caucasian", num_medications=-4), url)
    assert result.status == "invalid"
    assert "race" in result.message and "num_medications" in result.message
    assert "Caucasian" not in result.message and "-4" not in result.message


def test_client_reports_an_unloaded_model(serve):
    url = serve(unavailable)
    assert client.health(url).status == "model_unavailable"
    result = client.predict(valid_request(), url)
    assert result.status == "model_unavailable" and "store unreachable" in result.message


def test_client_reports_an_unreachable_api():
    url = f"http://127.0.0.1:{free_port()}"
    for result in (client.health(url, timeout=2), client.predict(valid_request(), url, timeout=2)):
        assert result.status == "unreachable" and url in result.message


def test_api_url_from_environment(monkeypatch):
    monkeypatch.delenv("READMISSION_API_URL", raising=False)
    assert client.api_url() == "http://localhost:8000"
    monkeypatch.setenv("READMISSION_API_URL", "http://api.example:9000/")
    assert client.api_url() == "http://api.example:9000"


def test_request_body_matches_the_api_schema():
    values = valid_request(**{drug: "No" for drug in config.MEDICATION_FEATURES})
    values.update(insulin="Steady", metformin="Up")
    body = client.request_body(values, other_medications=1)
    assert list(body) == config.FEATURE_COLUMNS
    assert body["medication_count"] == 3
    PredictionRequest.model_validate(body)


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

def run_page(monkeypatch, url):
    monkeypatch.setenv("READMISSION_API_URL", url)
    return AppTest.from_file(str(PAGE), default_timeout=60).run()


def test_page_works_without_the_api(monkeypatch):
    page = run_page(monkeypatch, f"http://127.0.0.1:{free_port()}")
    assert not page.exception
    assert "not clinically validated" in page.warning[0].value
    assert "not reachable" in page.error[0].value

    page.button[0].click().run()
    assert not page.exception
    assert any("No prediction" in error.value for error in page.error)
    assert len(page.metric) == 0


def test_page_shows_a_prediction(monkeypatch, serve):
    page = run_page(monkeypatch, serve(loader_for(0.25)))
    assert not page.exception
    assert "connected" in page.success[0].value

    page.button[0].click().run()
    assert not page.exception
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["Estimated 30-day readmission probability"] == "25.0%"
    assert metrics["Risk group"] == "Higher risk"


def test_page_reports_an_unloaded_model(monkeypatch, serve):
    page = run_page(monkeypatch, serve(unavailable))
    assert not page.exception
    assert "model is not loaded" in page.error[0].value


def test_page_offers_no_race_or_payer_and_no_excluded_discharge(monkeypatch):
    page = run_page(monkeypatch, f"http://127.0.0.1:{free_port()}")
    keys = {widget.key for widget in page.selectbox} | {widget.key for widget in page.number_input}
    assert keys.isdisjoint({"race", "payer_code"})
    assert set(config.CATEGORICAL_FEATURES) <= keys
    discharge = page.selectbox(key="discharge_disposition_id").options
    assert not any(option.split(":")[0] in {"11", "13", "14", "19", "20", "21"}
                   for option in discharge)
