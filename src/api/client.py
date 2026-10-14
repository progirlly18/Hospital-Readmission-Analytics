"""
Phase 8: a small client for the prediction API, used by the Streamlit page.

Standard library only, so the dashboard needs no extra dependency. Every
call returns an ApiResult instead of raising, so the page can show a clear
message when the API is down, the model is unavailable or an input is
rejected. Nothing is logged; the request body is sent to the API only.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from src.ml import config

DEFAULT_API_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 10


def api_url():
    """Base URL from READMISSION_API_URL, without a trailing slash."""
    return os.environ.get("READMISSION_API_URL", DEFAULT_API_URL).rstrip("/")


@dataclass
class ApiResult:
    # "ok", "invalid" (422), "model_unavailable" (503), "unreachable" or "error"
    status: str
    data: dict = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self):
        return self.status == "ok"


def _call(method, path, body=None, base_url=None, timeout=TIMEOUT_SECONDS):
    url = (base_url or api_url()) + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return ApiResult("ok", json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("detail")
        except (ValueError, AttributeError):
            detail = None
        if error.code == 422:
            return ApiResult("invalid", message=_validation_message(detail))
        if error.code == 503:
            return ApiResult("model_unavailable",
                             message=str(detail or "The model is not available."))
        return ApiResult("error", message=f"The API returned HTTP {error.code}.")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return ApiResult("unreachable",
                         message=f"The prediction API is not reachable at {url}.")
    except ValueError:
        return ApiResult("error", message="The API returned an unreadable response.")


def _validation_message(detail):
    """Field names and reasons from FastAPI's 422 body (inputs are not echoed)."""
    if not isinstance(detail, list):
        return str(detail or "The input was rejected.")
    reasons = []
    for item in detail:
        location = [str(part) for part in item.get("loc", []) if part != "body"]
        reasons.append(f"{'.'.join(location) or 'request'}: {item.get('msg', 'invalid')}")
    return "; ".join(reasons)


def health(base_url=None, timeout=TIMEOUT_SECONDS):
    return _call("GET", "/health", base_url=base_url, timeout=timeout)


def predict(features, base_url=None, timeout=TIMEOUT_SECONDS):
    return _call("POST", "/predict", body=features, base_url=base_url, timeout=timeout)


def request_body(values, other_medications=0):
    """
    The /predict body from the form's values for the 39 model inputs, with
    medication_count derived the way training does: the number of the 23
    diabetes medication columns not "No". 17 of them are model inputs and come
    from the form; other_medications counts the 6 that are not.
    """
    body = {column: values[column] for column in config.FEATURE_COLUMNS
            if column != "medication_count"}
    in_use = sum(values[column] != "No" for column in config.MEDICATION_FEATURES)
    body["medication_count"] = int(in_use + other_medications)
    return {column: body[column] for column in config.FEATURE_COLUMNS}
