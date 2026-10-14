"""
Phase 5: FastAPI inference service for the registered readmission model.

Run from the project root:

    uvicorn src.api.main:app

Endpoints:
    GET  /health      200 when the verified model is loaded, else 503
    GET  /model-info  model name, version, threshold, provenance, feature names
    POST /predict     one encounter's 39 features -> probability and flag
    GET  /metrics     aggregate request counts, failures, latency (Phase 7)

Request bodies and feature values are never logged or returned; validation
errors give the field and the problem, not the submitted value.
"""

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.ml import config
from src.api.model import (
    DISCLAIMER,
    POSITIVE_CLASS,
    ModelNotReady,
    Settings,
    load_model,
)
from src.api.telemetry import Telemetry
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


def _unavailable(app):
    return JSONResponse(
        status_code=503,
        content={"detail": f"Model unavailable: {app.state.model_error}"},
    )


def feature_frame(request):
    """One-row DataFrame with the dtypes used in training (build_feature_frame)."""
    frame = pd.DataFrame([request.to_feature_row()], columns=config.FEATURE_COLUMNS)
    frame["age_years"] = frame["age_years"].astype("float64")
    for column in config.NUMERIC_FEATURES:
        if column != "age_years":
            frame[column] = frame[column].astype("int64")
    for column in config.CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype(object)
    return frame


def create_app(settings=None, loader=load_model):
    """Build the app; the model is loaded once when the app starts."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.model, app.state.model_error = None, None
        try:
            app.state.model = loader(settings or Settings.from_env())
        except ModelNotReady as error:
            app.state.model_error = str(error)
        yield

    app = FastAPI(
        title="Hospital readmission risk (academic prototype)",
        description=DISCLAIMER,
        lifespan=lifespan,
    )
    app.state.telemetry = Telemetry()

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        started = time.perf_counter()
        status_code = 500  # kept if the endpoint raises
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            app.state.telemetry.record_request(
                request.url.path, status_code, (time.perf_counter() - started) * 1000)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Only where and what; never the submitted value ("input").
        errors = [
            {"loc": list(error.get("loc", ())), "msg": error.get("msg", ""),
             "type": error.get("type", "")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health", response_model=HealthResponse)
    def health():
        loaded = app.state.model
        if loaded is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(status="unavailable", model_loaded=False,
                                       detail=app.state.model_error).model_dump(),
            )
        return HealthResponse(status="ok", model_loaded=True, model_version=loaded.version)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info():
        loaded = app.state.model
        if loaded is None:
            return _unavailable(app)
        return ModelInfoResponse(
            model_name=loaded.name,
            model_version=loaded.version,
            alias=loaded.alias,
            threshold=loaded.threshold,
            model_sha256=loaded.model_sha256,
            serialization_format=loaded.serialization_format,
            positive_class=POSITIVE_CLASS,
            features=list(loaded.feature_columns),
            disclaimer=DISCLAIMER,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        loaded = app.state.model
        if loaded is None:
            return _unavailable(app)
        probability = float(loaded.model.predict_proba(feature_frame(request))[0, 1])
        high_risk = probability >= loaded.threshold
        app.state.telemetry.record_prediction(probability, high_risk)
        return PredictionResponse(
            readmission_probability=probability,
            high_risk=high_risk,
            threshold=loaded.threshold,
            model_name=loaded.name,
            model_version=loaded.version,
            disclaimer=DISCLAIMER,
        )

    @app.get("/metrics")
    def metrics():
        return app.state.telemetry.snapshot(
            model_loaded=app.state.model is not None,
            model_error=app.state.model_error,
        )

    return app


app = create_app()
