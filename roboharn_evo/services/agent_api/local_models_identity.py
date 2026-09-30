#!/usr/bin/env python3
"""Build a stable identity from an OpenAI-compatible ``/v1/models`` payload.

vLLM regenerates ``created`` timestamps and permission IDs for every request.
Those fields are useful response metadata, but they cannot identify a running
model service across two otherwise equivalent requests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


_STABLE_MODEL_FIELDS = (
    "id",
    "object",
    "owned_by",
    "root",
    "parent",
    "max_model_len",
)
_STABLE_PERMISSION_FIELDS = (
    "object",
    "allow_create_engine",
    "allow_sampling",
    "allow_logprobs",
    "allow_search_indices",
    "allow_view",
    "allow_fine_tuning",
    "organization",
    "group",
    "is_blocking",
)


class ModelsIdentityError(ValueError):
    """Raised when a models payload cannot provide a stable model identity."""


def canonical_models_identity(payload: Any) -> dict[str, Any]:
    """Return stable model-card fields, excluding per-request metadata."""

    if not isinstance(payload, dict):
        raise ModelsIdentityError("models response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelsIdentityError("models response data must be a list")

    models: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ModelsIdentityError(f"models response data[{index}] must be an object")
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ModelsIdentityError(
                f"models response data[{index}].id must be a non-empty string"
            )
        stable = {field: item.get(field) for field in _STABLE_MODEL_FIELDS}
        permissions = item.get("permission", [])
        if not isinstance(permissions, list):
            raise ModelsIdentityError(
                f"models response data[{index}].permission must be a list"
            )
        stable_permissions: list[dict[str, Any]] = []
        for permission_index, permission in enumerate(permissions):
            if not isinstance(permission, dict):
                raise ModelsIdentityError(
                    "models response "
                    f"data[{index}].permission[{permission_index}] must be an object"
                )
            stable_permissions.append(
                {field: permission.get(field) for field in _STABLE_PERMISSION_FIELDS}
            )
        stable["permission"] = sorted(
            stable_permissions,
            key=lambda value: json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
        models.append(stable)

    models.sort(
        key=lambda value: (
            str(value.get("id")),
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    )
    return {"object": payload.get("object"), "data": models}


def models_identity_sha256(payload: Any) -> str:
    canonical = canonical_models_identity(payload)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
