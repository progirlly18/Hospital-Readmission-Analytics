"""
Load and verify the registered model for the prediction API.

The model is resolved once, at startup, from the MLflow registry alias
(@candidate). Before it may serve predictions, every check below must pass;
otherwise the API reports itself unavailable (HTTP 503):

- the alias resolves to a READY version, which is then loaded by its version
  number so the alias cannot move between the checks and the load;
- the stored model uses the skops format (no pickle-based loading);
- the signature's inputs are exactly config.FEATURE_COLUMNS, in order, and
  the output is the two-column predict_proba array;
- the decision threshold in the version tag and in the MLmodel metadata
  equals EXPECTED_THRESHOLD;
- the model file hash in the version tag, the MLmodel metadata and the source
  run tag equals TRUSTED_MODEL_SHA256 (the Phase 3 file verified in Phase 4),
  and the version is labelled as the unchanged Phase 3 model.

Inside a container the registry is not reachable, so the model can instead be
loaded from a bundle exported from it (src/api/bundle.py). The bundle goes
through the same checks, plus a hash check of every exported file.
"""

import os
from dataclasses import dataclass, field

# Imported before mlflow: src.ml.tracking sets MLflow's environment options.
from src.ml import config
from src.ml.register_model import MODEL_ALIAS, RECORD_TYPE, REGISTERED_MODEL_NAME
from src.ml.tracking import TRACKING_URI

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402


EXPECTED_THRESHOLD = 0.1617830043681171
TRUSTED_MODEL_SHA256 = "1d17bf4ac345891a58b4566ca5b3265798fea2cc73aa2dde860015dfc25062f9"
# model.skops of registered version 1 (the same model, as stored by MLflow).
TRUSTED_ARTIFACT_SHA256 = "13f79fe55a2112ad17fc77af4b8cda65784a090fb9890bb8b50688f1a704d1d5"
DISCLAIMER = (
    "Academic decision-support prototype; not clinically validated and not "
    "for real patient decisions."
)
POSITIVE_CLASS = "readmitted <30 days"


class ModelNotReady(Exception):
    """The registered model could not be loaded or failed verification."""


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = TRACKING_URI
    model_name: str = REGISTERED_MODEL_NAME
    alias: str = MODEL_ALIAS
    expected_threshold: float = EXPECTED_THRESHOLD
    trusted_sha256: str = TRUSTED_MODEL_SHA256
    trusted_artifact_sha256: str = TRUSTED_ARTIFACT_SHA256
    feature_columns: tuple = field(default_factory=lambda: tuple(config.FEATURE_COLUMNS))
    bundle_dir: str | None = None

    @classmethod
    def from_env(cls):
        """Only the store or bundle location can be changed from the environment."""
        return cls(tracking_uri=os.environ.get("READMISSION_TRACKING_URI", TRACKING_URI),
                   bundle_dir=os.environ.get("READMISSION_MODEL_BUNDLE") or None)


@dataclass(frozen=True)
class LoadedModel:
    model: object
    name: str
    version: str
    alias: str
    threshold: float
    model_sha256: str
    serialization_format: str
    feature_columns: tuple


def verify(settings, version, info, run_tags):
    """
    Check one registered version before loading it; raise ModelNotReady with
    the first problem found. `info` is mlflow.models.get_model_info output.
    """
    if version.status != "READY":
        raise ModelNotReady(f"model version {version.version} is not READY")

    sklearn_flavor = info.flavors.get("sklearn", {})
    if sklearn_flavor.get("serialization_format") != "skops":
        raise ModelNotReady("model is not stored in the skops format; refusing to load")

    if info.signature is None:
        raise ModelNotReady("model has no signature")
    names = tuple(column["name"] for column in info.signature.inputs.to_dict())
    if names != settings.feature_columns:
        raise ModelNotReady("model input names or order differ from the 39 expected features")
    outputs = info.signature.outputs.to_dict()
    if len(outputs) != 1 or tuple(outputs[0].get("tensor-spec", {}).get("shape", ())) != (-1, 2):
        raise ModelNotReady("model output is not a two-column probability array")

    tags, metadata = version.tags, info.metadata or {}
    try:
        tag_threshold = float(tags["selected_threshold"])
        metadata_threshold = float(metadata["selected_threshold"])
    except (KeyError, TypeError, ValueError):
        raise ModelNotReady("decision threshold is missing from the model version") from None
    if not tag_threshold == metadata_threshold == settings.expected_threshold:
        raise ModelNotReady("decision threshold does not match the expected value")

    hashes = (tags.get("model_sha256"), metadata.get("model_sha256"),
              run_tags.get("model_sha256"))
    if any(value != settings.trusted_sha256 for value in hashes):
        raise ModelNotReady("model provenance hash does not match the trusted Phase 3 model")
    if tags.get("record_type") != RECORD_TYPE or tags.get("retrained") != "false":
        raise ModelNotReady("model version is not labelled as the unchanged Phase 3 model")

    return tag_threshold


def resolve(settings):
    """Look up the alias in the registry: (version, model uri, model info, run tags)."""
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    mlflow.set_tracking_uri(settings.tracking_uri)
    try:
        version = client.get_model_version_by_alias(settings.model_name, settings.alias)
        uri = f"models:/{settings.model_name}/{version.version}"
        info = mlflow.models.get_model_info(uri)
        run_tags = client.get_run(version.run_id).data.tags
    except Exception as error:  # store unreachable, alias or version missing
        raise ModelNotReady(
            f"registered model {settings.model_name}@{settings.alias} is unavailable "
            f"({type(error).__name__})"
        ) from None
    return version, uri, info, run_tags


def load_verified(settings, version, threshold, uri):
    """Load a model that passed verify() and check it is a binary classifier."""
    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception as error:
        raise ModelNotReady(f"model could not be loaded ({type(error).__name__})") from None
    if list(getattr(model, "classes_", [])) != [0, 1] or not hasattr(model, "predict_proba"):
        raise ModelNotReady("loaded model is not a binary probability classifier")

    return LoadedModel(
        model=model,
        name=settings.model_name,
        version=str(version.version),
        alias=settings.alias,
        threshold=threshold,
        model_sha256=version.tags["model_sha256"],
        serialization_format="skops",
        feature_columns=settings.feature_columns,
    )


def load_model(settings):
    """Resolve, verify and load the model (registry or bundle); raise ModelNotReady."""
    if settings.bundle_dir:
        from src.api.bundle import load_bundle
        return load_bundle(settings)

    version, uri, info, run_tags = resolve(settings)
    threshold = verify(settings, version, info, run_tags)
    return load_verified(settings, version, threshold, uri)
