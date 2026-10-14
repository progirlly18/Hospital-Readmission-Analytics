"""
Request and response schemas for the prediction API.

The request has exactly the 39 model features (config.FEATURE_COLUMNS), all
required, in the types the registered model's signature expects:

- age_years: number, one of the 10-year band midpoints 5, 15, ..., 95
- the other 9 numeric features: integers (no strings, floats or booleans)
- the 29 categorical features: strings from ALLOWED_VALUES

Unknown fields (identifiers, race, payer_code, ...) are rejected. Expired and
hospice discharge codes are rejected with their own message because those
encounters are outside the model's cohort (Phase 2).
"""

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    create_model,
)
from pydantic_core import PydanticCustomError

from src.ml import config
from src.api.allowed_values import ALLOWED_VALUES


AGE_BAND_MIDPOINTS = tuple(float(age) for age in range(5, 100, 10))
EXCLUDED_DISCHARGE_CODES = tuple(sorted(
    str(code) for code in config.EXPIRED_DISPOSITION_IDS | config.HOSPICE_DISPOSITION_IDS
))

# Integer bounds: the dataset's own limits where it has one (stay of 1-14
# days, 23 medication columns), otherwise non-negative with a sanity cap.
INTEGER_BOUNDS = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (0, 1000),
    "num_procedures": (0, 1000),
    "num_medications": (0, 1000),
    "number_outpatient": (0, 1000),
    "number_emergency": (0, 1000),
    "number_inpatient": (0, 1000),
    "number_diagnoses": (0, 1000),
    "medication_count": (0, len(config.ALL_MEDICATION_COLUMNS)),
}


def _age_band_midpoint(value):
    if value not in AGE_BAND_MIDPOINTS:
        raise PydanticCustomError(
            "age_band",
            "age_years must be a 10-year age band midpoint: 5, 15, ..., 95",
        )
    return float(value)


def _not_excluded_discharge(value):
    if isinstance(value, str) and value in EXCLUDED_DISCHARGE_CODES:
        raise PydanticCustomError(
            "excluded_discharge",
            "Expired or hospice discharge codes ({codes}) are outside the "
            "model's cohort; no prediction is made for them.",
            {"codes": ", ".join(EXCLUDED_DISCHARGE_CODES)},
        )
    return value


def _field_type(column):
    if column == "age_years":
        return Annotated[float, AfterValidator(_age_band_midpoint)]
    if column in INTEGER_BOUNDS:
        low, high = INTEGER_BOUNDS[column]
        return Annotated[int, Field(ge=low, le=high)]
    allowed = Literal[ALLOWED_VALUES[column]]
    if column == "discharge_disposition_id":
        return Annotated[allowed, BeforeValidator(_not_excluded_discharge)]
    return allowed


def _python_name(column):
    """Medication pairs such as "glyburide-metformin" are not identifiers."""
    return column.replace("-", "_")


class _RequestBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, populate_by_name=False,
        hide_input_in_errors=True,
    )

    def to_feature_row(self):
        """The validated values keyed by model column name, in model order."""
        return {column: getattr(self, _python_name(column))
                for column in config.FEATURE_COLUMNS}


# Built from config.FEATURE_COLUMNS so the fields and their order are exactly
# the model's 39 inputs; the JSON keys are the model's column names.
PredictionRequest = create_model(
    "PredictionRequest",
    __base__=_RequestBase,
    **{
        _python_name(column): (_field_type(column), Field(alias=column))
        for column in config.FEATURE_COLUMNS
    },
)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    readmission_probability: float
    high_risk: bool
    threshold: float
    model_name: str
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    alias: str
    threshold: float
    model_sha256: str
    serialization_format: str
    positive_class: str
    features: list[str]
    disclaimer: str
