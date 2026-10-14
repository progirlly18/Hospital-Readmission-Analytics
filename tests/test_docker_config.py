"""Static checks of the Phase 6 container configuration (no Docker needed)."""

import re

from src.ml import config

ROOT = config.PROJECT_ROOT


def requirement_pins(name):
    pins = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            package, version = line.split("==")
            pins[package.lower()] = version
    return pins


def dockerfile_instructions():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    text = text.replace("\\\n", " ")
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def test_api_requirements_are_pinned_like_the_ml_requirements():
    api, ml = requirement_pins("requirements-api.txt"), requirement_pins("requirements-ml.txt")
    for package, version in api.items():
        if package in ml:
            assert ml[package] == version, package
    assert api["mlflow-skinny"] == ml["mlflow"]
    assert {"fastapi", "uvicorn", "skops", "scikit-learn"} <= set(api)
    assert not {"pytest", "xgboost", "matplotlib", "httpx"} & set(api)


def test_dockerfile_copies_code_only_and_runs_as_non_root():
    instructions = dockerfile_instructions()
    copies = [line for line in instructions if line.startswith("COPY")]
    assert copies == ["COPY requirements-api.txt .", "COPY src/ src/"]
    assert re.match(r"FROM python:3\.11\.\d+-slim", instructions[0])
    users = [line for line in instructions if line.startswith("USER")]
    assert users and users[-1] not in ("USER root", "USER 0")
    assert "EXPOSE 8000" in instructions
    assert any("READMISSION_MODEL_BUNDLE=/model-bundle" in line for line in instructions)
    assert not any(re.search(r"(?i)(secret|token|password|api_key)", line)
                   for line in instructions)


def test_dockerignore_is_an_allow_list_that_keeps_data_and_store_out():
    lines = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert lines[0] == "*"
    allowed = [line[1:] for line in lines if line.startswith("!")]
    assert "src/api/" in allowed and "requirements-api.txt" in allowed
    for private in ("data", "mlflow.db", "mlartifacts", "artifacts", "model_bundle",
                    ".env", "dags", "dashboard", "sql"):
        assert not any(entry.startswith(private) for entry in allowed), private
