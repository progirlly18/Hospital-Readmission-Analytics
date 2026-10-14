import json
import logging
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml import config
from src.ml.tracking import setup_experiment, sha256_file

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.allowed_values import ALLOWED_VALUES  # noqa: E402
from src.api.main import create_app  # noqa: E402
from src.api.model import (  # noqa: E402
    EXPECTED_THRESHOLD,
    TRUSTED_MODEL_SHA256,
    ModelNotReady,
    Settings,
    verify,
)
from src.api.schemas import EXCLUDED_DISCHARGE_CODES  # noqa: E402
from src.ml.log_phase3_to_mlflow import RECORD_TYPE as PHASE3_RECORD_TYPE  # noqa: E402
from src.ml.register_model import (  # noqa: E402
    PHASE3_PARENT_RUN_NAME,
    RECORD_TYPE,
    load_verified_model,
    register,
)


GOOD_MODEL = "api-test-model"
WRONG_THRESHOLD_MODEL = "api-test-wrong-threshold"


# ---------------------------------------------------------
# Synthetic data and model (fixtures only; no project data)
# ---------------------------------------------------------

def valid_request(**changes):
    """One valid request body: every field present with an allowed value."""
    body = {column: ALLOWED_VALUES[column][0] for column in config.CATEGORICAL_FEATURES}
    body.update({
        "age_years": 65.0, "time_in_hospital": 4, "num_lab_procedures": 40,
        "num_procedures": 1, "num_medications": 15, "number_outpatient": 0,
        "number_emergency": 0, "number_inpatient": 1, "number_diagnoses": 9,
        "medication_count": 2,
        "gender": "Female", "discharge_disposition_id": "1",
        "medical_specialty": "Cardiology",
    })
    body.update(changes)
    return body


def synthetic_rows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = {column: rng.choice(ALLOWED_VALUES[column], n).astype(object)
            for column in config.CATEGORICAL_FEATURES}
    data["age_years"] = rng.choice(np.arange(5.0, 100.0, 10.0), n)
    for column in config.NUMERIC_FEATURES[1:]:
        data[column] = rng.integers(1, 10, n)
    return pd.DataFrame(data)[config.FEATURE_COLUMNS]


def synthetic_model(rows):
    y = (rows["number_inpatient"] + np.random.default_rng(1).normal(0, 2, len(rows)) > 5)
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             config.CATEGORICAL_FEATURES),
        ], remainder="passthrough")),
        ("model", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)),
    ])
    return CalibratedClassifierCV(pipeline, method="sigmoid", cv=3).fit(rows, y.astype(int))


def register_synthetic(root, uri, name, threshold=EXPECTED_THRESHOLD):
    """Save, record and register a synthetic model the way Phase 3/4 do."""
    rows = synthetic_rows()
    phase3 = root / name
    (phase3 / "models").mkdir(parents=True)
    joblib.dump(synthetic_model(rows), phase3 / "models" / "random_forest.joblib")
    (phase3 / "metrics.json").write_text(json.dumps({
        "split_fingerprint_sha256": "ab" * 32,
        "positive_class": "readmitted <30 days",
        "selected_model": "random_forest",
        "models": {"random_forest": {"selected_threshold": threshold}},
        "test_selected_model": {"roc_auc": 0.67},
    }), encoding="utf-8")
    model_sha256 = sha256_file(phase3 / "models" / "random_forest.joblib")
    experiment_id = setup_experiment(name, uri, root / "mlartifacts")
    with mlflow.start_run(experiment_id=experiment_id, run_name=PHASE3_PARENT_RUN_NAME,
                          tags={"record_type": PHASE3_RECORD_TYPE,
                                "metrics_json_sha256": sha256_file(phase3 / "metrics.json"),
                                "selected_model_sha256": model_sha256}):
        pass
    register(phase3, uri, name, root / "mlartifacts", name, rows=rows)
    return model_sha256


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A temporary MLflow store with two synthetic registered models."""
    root = tmp_path_factory.mktemp("api")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    good_sha = register_synthetic(root, uri, GOOD_MODEL)
    wrong_sha = register_synthetic(root, uri, WRONG_THRESHOLD_MODEL, threshold=0.3)
    yield {"uri": uri, "good_sha": good_sha, "wrong_sha": wrong_sha}
    mlflow.set_tracking_uri(None)


def settings_for(store, **changes):
    values = {"tracking_uri": store["uri"], "model_name": GOOD_MODEL,
              "trusted_sha256": store["good_sha"], **changes}
    return Settings(**values)


@pytest.fixture(scope="module")
def client(store):
    with TestClient(create_app(settings_for(store))) as test_client:
        yield test_client


def assert_422_on(response, field):
    assert response.status_code == 422
    assert [field] in [error["loc"][1:] for error in response.json()["detail"]]


# ---------------------------------------------------------
# Valid requests
# ---------------------------------------------------------

def test_health_and_model_info_when_verified(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok" and health.json()["model_version"] == "1"
    info = client.get("/model-info").json()
    assert info["threshold"] == EXPECTED_THRESHOLD
    assert info["features"] == config.FEATURE_COLUMNS
    assert info["serialization_format"] == "skops"
    assert info["alias"] == "candidate"


def test_valid_request_returns_probability_and_flag(client):
    response = client.post("/predict", json=valid_request())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"readmission_probability", "high_risk", "threshold",
                         "model_name", "model_version", "disclaimer"}
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["threshold"] == EXPECTED_THRESHOLD
    assert body["high_risk"] == (body["readmission_probability"] >= EXPECTED_THRESHOLD)


def test_prediction_equals_the_loaded_model(client):
    model = client.app.state.model.model
    for seed in range(3):
        row = synthetic_rows(n=1, seed=seed + 10)
        body = {k: (float(v) if k == "age_years" else int(v)
                    if k in config.NUMERIC_FEATURES else v)
                for k, v in row.iloc[0].items()}
        body["discharge_disposition_id"] = "1"
        row["discharge_disposition_id"] = "1"
        response = client.post("/predict", json=body)
        assert response.status_code == 200
        assert response.json()["readmission_probability"] == pytest.approx(
            model.predict_proba(row)[0, 1])


def test_flag_uses_greater_or_equal_at_the_threshold(client, monkeypatch):
    model = client.app.state.model.model
    monkeypatch.setattr(model, "predict_proba",
                        lambda frame: np.array([[1 - EXPECTED_THRESHOLD, EXPECTED_THRESHOLD]]))
    assert client.post("/predict", json=valid_request()).json()["high_risk"] is True


def test_integer_age_is_accepted(client):
    assert client.post("/predict", json=valid_request(age_years=65)).status_code == 200


def test_no_feature_values_in_response_or_logs(client, caplog):
    marker = "Perinatology"  # a rare allowed specialty, easy to search for
    with caplog.at_level(logging.DEBUG):
        response = client.post("/predict", json=valid_request(medical_specialty=marker))
    assert response.status_code == 200
    for text in (response.text, caplog.text):
        assert marker not in text
        assert "Female" not in text


# ---------------------------------------------------------
# Invalid requests (HTTP 422)
# ---------------------------------------------------------

def test_missing_field(client):
    body = valid_request()
    del body["insulin"]
    assert_422_on(client.post("/predict", json=body), "insulin")


@pytest.mark.parametrize("field, value", [
    ("race", "Caucasian"), ("encounter_id", "12345"), ("patient_nbr", "1"),
    ("payer_code", "MC"), ("readmitted", "<30"),
])
def test_extra_fields_are_rejected_without_echoing_them(client, field, value):
    response = client.post("/predict", json=valid_request(**{field: value}))
    assert_422_on(response, field)
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    assert value not in response.text


@pytest.mark.parametrize("field, value", [
    ("time_in_hospital", "5"), ("time_in_hospital", 5.5), ("time_in_hospital", 5.0),
    ("number_inpatient", True), ("age_years", "65"), ("age_years", 66),
    ("admission_type_id", 1), ("gender", None), ("medical_specialty", None),
])
def test_wrong_types_are_rejected(client, field, value):
    assert_422_on(client.post("/predict", json=valid_request(**{field: value})), field)


@pytest.mark.parametrize("field, value", [
    ("time_in_hospital", 0), ("time_in_hospital", 15), ("num_medications", -1),
    ("medication_count", 24),
])
def test_out_of_range_numbers_are_rejected(client, field, value):
    assert_422_on(client.post("/predict", json=valid_request(**{field: value})), field)


@pytest.mark.parametrize("field, value", [
    ("medical_specialty", "Astrology"), ("medical_specialty", "Unknown"),
    ("gender", "Unknown/Invalid"), ("diag_1_group", "Other"),
    ("A1Cresult", ">9"), ("insulin", "Sometimes"), ("discharge_disposition_id", "99"),
])
def test_invalid_categories_are_rejected_without_echoing_them(client, field, value):
    response = client.post("/predict", json=valid_request(**{field: value}))
    assert_422_on(response, field)
    assert value not in response.text


@pytest.mark.parametrize("code", EXCLUDED_DISCHARGE_CODES)
def test_expired_and_hospice_discharges_are_rejected(client, code):
    response = client.post("/predict", json=valid_request(discharge_disposition_id=code))
    assert_422_on(response, "discharge_disposition_id")
    assert response.json()["detail"][0]["type"] == "excluded_discharge"
    assert "outside the model's cohort" in response.json()["detail"][0]["msg"]


def test_excluded_codes_are_not_allowed_values():
    assert set(EXCLUDED_DISCHARGE_CODES) == {"11", "13", "14", "19", "20", "21"}
    assert not set(EXCLUDED_DISCHARGE_CODES) & set(ALLOWED_VALUES["discharge_disposition_id"])


def test_malformed_json_is_rejected(client):
    response = client.post("/predict", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 422


# ---------------------------------------------------------
# Model unavailable or not verified (HTTP 503)
# ---------------------------------------------------------

def assert_unavailable(app_settings, reason):
    with TestClient(create_app(app_settings)) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert reason in health.json()["detail"]
        assert test_client.get("/model-info").status_code == 503
        predict = test_client.post("/predict", json=valid_request())
        assert predict.status_code == 503
        assert reason in predict.json()["detail"]


def test_missing_registered_model_gives_503(store):
    assert_unavailable(settings_for(store, model_name="no-such-model"), "unavailable")


def test_missing_alias_gives_503(store):
    assert_unavailable(settings_for(store, alias="production"), "unavailable")


def test_untrusted_hash_gives_503(store):
    assert_unavailable(settings_for(store, trusted_sha256="0" * 64), "provenance hash")


def test_wrong_threshold_gives_503(store):
    assert_unavailable(
        settings_for(store, model_name=WRONG_THRESHOLD_MODEL,
                     trusted_sha256=store["wrong_sha"]),
        "decision threshold",
    )


def test_loader_failure_gives_503():
    def failing_loader(settings):
        raise ModelNotReady("store unreachable")
    with TestClient(create_app(Settings(), loader=failing_loader)) as test_client:
        assert test_client.get("/health").status_code == 503
        assert test_client.post("/predict", json=valid_request()).status_code == 503


def test_invalid_request_is_422_even_when_model_unavailable():
    def failing_loader(settings):
        raise ModelNotReady("store unreachable")
    with TestClient(create_app(Settings(), loader=failing_loader)) as test_client:
        body = valid_request(race="x")
        assert test_client.post("/predict", json=body).status_code == 422


# ---------------------------------------------------------
# verify(): each provenance and schema check on its own
# ---------------------------------------------------------

def fake_registry_entry(**changes):
    """A version/info/run-tags triple that passes verify(); `changes` breaks one part."""
    inputs = [{"name": c, "type": "string", "required": True} for c in config.FEATURE_COLUMNS]
    outputs = [{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 2]}}]
    parts = {
        "status": "READY",
        "tags": {"selected_threshold": repr(EXPECTED_THRESHOLD), "model_sha256": TRUSTED_MODEL_SHA256,
                 "record_type": RECORD_TYPE, "retrained": "false"},
        "metadata": {"selected_threshold": EXPECTED_THRESHOLD, "model_sha256": TRUSTED_MODEL_SHA256},
        "format": "skops", "inputs": inputs, "outputs": outputs,
        "run_tags": {"model_sha256": TRUSTED_MODEL_SHA256},
    }
    parts.update(changes)
    version = SimpleNamespace(status=parts["status"], version="1", tags=parts["tags"])
    signature = SimpleNamespace(inputs=SimpleNamespace(to_dict=lambda: parts["inputs"]),
                                outputs=SimpleNamespace(to_dict=lambda: parts["outputs"]))
    info = SimpleNamespace(flavors={"sklearn": {"serialization_format": parts["format"]}},
                           signature=signature, metadata=parts["metadata"])
    return version, info, parts["run_tags"]


def test_verify_accepts_the_expected_entry():
    assert verify(Settings(), *fake_registry_entry()) == EXPECTED_THRESHOLD


@pytest.mark.parametrize("changes, reason", [
    ({"status": "PENDING_REGISTRATION"}, "not READY"),
    ({"format": "cloudpickle"}, "skops"),
    ({"format": None}, "skops"),
    ({"inputs": [{"name": c} for c in reversed(config.FEATURE_COLUMNS)]}, "order"),
    ({"inputs": [{"name": c} for c in config.FEATURE_COLUMNS[:-1]]}, "order"),
    ({"outputs": [{"type": "tensor", "tensor-spec": {"shape": [-1]}}]}, "two-column"),
    ({"tags": {"model_sha256": TRUSTED_MODEL_SHA256}}, "threshold is missing"),
    ({"metadata": {"selected_threshold": 0.16, "model_sha256": TRUSTED_MODEL_SHA256}},
     "threshold does not match"),
    ({"run_tags": {"model_sha256": "0" * 64}}, "provenance hash"),
    ({"run_tags": {}}, "provenance hash"),
    ({"tags": {"selected_threshold": repr(EXPECTED_THRESHOLD), "model_sha256": TRUSTED_MODEL_SHA256,
               "record_type": RECORD_TYPE, "retrained": "true"}}, "unchanged Phase 3"),
])
def test_verify_rejects_each_failure(changes, reason):
    with pytest.raises(ModelNotReady, match=reason):
        verify(Settings(), *fake_registry_entry(**changes))


# ---------------------------------------------------------
# Checks against the real project files (skipped if absent)
# ---------------------------------------------------------

REAL_MODEL = config.ARTIFACTS_DIR / "phase3" / "models" / "random_forest.joblib"
REAL_STORE = config.PROJECT_ROOT / "mlflow.db"


@pytest.mark.skipif(not config.CLEANED_FILE.is_file(), reason="cleaned data not available")
def test_allowed_values_equal_the_cohort_values():
    from src.ml.data import load_modelling_data
    from src.ml.features import build_feature_frame
    cohort, _ = load_modelling_data()
    frame = build_feature_frame(cohort)
    for column in config.CATEGORICAL_FEATURES:
        assert tuple(sorted(frame[column].astype(str).unique())) == ALLOWED_VALUES[column], column
    assert len(ALLOWED_VALUES["medical_specialty"]) == 73


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="Phase 3 model not available")
def test_model_categories_are_all_allowed():
    """Every category the saved model learned is accepted by the API."""
    model, _ = load_verified_model(REAL_MODEL, TRUSTED_MODEL_SHA256)
    preprocess = model.calibrated_classifiers_[0].estimator.named_steps["preprocess"]
    encoder = preprocess.named_transformers_["categorical"].named_steps["encode"]
    columns = preprocess.transformers_[1][2]
    for column, categories in zip(columns, encoder.categories_):
        assert {str(c) for c in categories} <= set(ALLOWED_VALUES[column]), column


@pytest.mark.skipif(not REAL_STORE.is_file(), reason="project MLflow store not available")
def test_real_registered_model_is_served_read_only():
    with TestClient(create_app(Settings())) as test_client:
        health = test_client.get("/health").json()
        assert health["status"] == "ok" and health["model_version"] == "1"
        info = test_client.get("/model-info").json()
        assert info["model_sha256"] == TRUSTED_MODEL_SHA256
        assert info["threshold"] == EXPECTED_THRESHOLD
        body = test_client.post("/predict", json=valid_request()).json()
        assert 0.0 <= body["readmission_probability"] <= 1.0
