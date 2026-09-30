from __future__ import annotations

import re
from typing import Any


ROLE_PRIORITY = {"target": 0, "tool": 1, "context": 2}
ENTITY_SCOPES = {"single_instance", "reference_set", "robot_state"}
PLACEMENT_RELATIONS = {"center_of"}
REFERENCE_QUERY_METADATA_KEYS = (
    "entity_scope",
    "placement_relation",
    "expected_count",
)


def normalize_perception_object_id(value: Any) -> tuple[str, str]:
    """Return a schema-safe object_id without semantic interpretation.

    This intentionally does not split descriptors such as "left", "brown", or
    "first" out of the object id. Semantic normalization belongs to the VLM
    normalization skill/API, not Python keyword lists.
    """
    return sanitize_token_text(value), ""


def merge_instance_hints(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        text = sanitize_hint_text(value)
        if text:
            parts.append(text)
    return "; ".join(dedupe_preserving_order(parts))


def normalize_entity_scope(value: Any) -> str:
    scope = sanitize_token_text(value)
    return scope if scope in ENTITY_SCOPES else "single_instance"


def normalize_placement_relation(value: Any) -> str:
    relation = sanitize_token_text(value)
    return relation if relation in PLACEMENT_RELATIONS else ""


def normalize_expected_count(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if 1 <= count <= 32 else None


def normalized_relation_metadata(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    relation = normalize_placement_relation(
        payload.get("placement_relation")
    )
    scope = normalize_entity_scope(payload.get("entity_scope"))
    if relation:
        result: dict[str, Any] = {
            "entity_scope": "reference_set",
            "placement_relation": relation,
        }
        expected_count = normalize_expected_count(
            payload.get("expected_count")
        )
        if expected_count is not None:
            result["expected_count"] = expected_count
        return result
    return {"entity_scope": scope} if scope != "single_instance" else {}


def merge_perception_query(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    existing_role = normalize_query_role(merged.get("role", "context"))
    incoming_role = normalize_query_role(incoming.get("role", "context"))
    incoming_priority = ROLE_PRIORITY[incoming_role]
    existing_priority = ROLE_PRIORITY[existing_role]
    merged["role"] = incoming_role if incoming_priority < existing_priority else existing_role

    object_id = str(merged.get("object_id") or incoming.get("object_id") or "").strip()
    merged["object_id"] = object_id
    current_prompt = str(merged.get("text_prompt", "")).strip()
    incoming_prompt = str(incoming.get("text_prompt", "")).strip()
    if incoming_priority < existing_priority and incoming_prompt:
        merged["text_prompt"] = incoming_prompt
    elif not current_prompt and incoming_prompt:
        merged["text_prompt"] = incoming_prompt
    else:
        merged["text_prompt"] = current_prompt or object_id
    if incoming_priority < existing_priority:
        merged["instance_hint"] = sanitize_hint_text(incoming.get("instance_hint", "")) or sanitize_hint_text(merged.get("instance_hint", ""))
    elif incoming_priority == existing_priority:
        merged["instance_hint"] = merge_instance_hints(merged.get("instance_hint", ""), incoming.get("instance_hint", ""))
    else:
        merged["instance_hint"] = sanitize_hint_text(merged.get("instance_hint", ""))
    merged["reason"] = merge_reason_text(merged.get("reason", ""), incoming.get("reason", ""))
    existing_relation = normalize_placement_relation(
        merged.get("placement_relation")
    )
    incoming_relation = normalize_placement_relation(
        incoming.get("placement_relation")
    )
    relation = incoming_relation or existing_relation
    if relation:
        merged["entity_scope"] = "reference_set"
        merged["placement_relation"] = relation
        expected_counts = [
            count
            for count in (
                normalize_expected_count(merged.get("expected_count")),
                normalize_expected_count(incoming.get("expected_count")),
            )
            if count is not None
        ]
        if expected_counts:
            merged["expected_count"] = max(expected_counts)
    return merged


def matching_perception_query_index(
    queries: list[dict[str, Any]],
    incoming: dict[str, Any],
) -> int | None:
    """Find an exact or generic duplicate without collapsing distinct instances.

    Bound queries are identities and therefore merge only on the advertised
    identity fields.  Before candidates exist, ``instance_hint`` is the only
    schema-level instance discriminator.  Two non-empty, different hints must
    survive as separate segmentation queries; a single generic query may still
    coalesce with its one specific counterpart so exact category duplicates do
    not consume the bounded query budget.
    """
    object_id = str(incoming.get("object_id", "") or "").strip().lower()
    instance_ref = str(incoming.get("instance_ref", "") or "").strip()
    oracle_id = str(incoming.get("oracle_id", "") or "").strip()
    incoming_hint = sanitize_hint_text(incoming.get("instance_hint", ""))
    incoming_relation = normalize_placement_relation(
        incoming.get("placement_relation")
    )
    same_category_unbound: list[tuple[int, str]] = []
    for index, existing in enumerate(queries):
        if str(existing.get("object_id", "") or "").strip().lower() != object_id:
            continue
        existing_ref = str(existing.get("instance_ref", "") or "").strip()
        existing_oracle = str(existing.get("oracle_id", "") or "").strip()
        if instance_ref or oracle_id or existing_ref or existing_oracle:
            if (
                instance_ref == existing_ref
                and oracle_id == existing_oracle
            ):
                return index
            continue
        existing_hint = sanitize_hint_text(
            existing.get("instance_hint", "")
        )
        if normalize_placement_relation(
            existing.get("placement_relation")
        ) != incoming_relation:
            continue
        if incoming_hint and existing_hint == incoming_hint:
            return index
        same_category_unbound.append((index, existing_hint))
    if not same_category_unbound:
        return None
    if incoming_hint:
        generic = [
            index for index, existing_hint in same_category_unbound
            if not existing_hint
        ]
        if len(generic) == 1 and len(same_category_unbound) == 1:
            return generic[0]
        return None
    # An unqualified category query is redundant with an already retained
    # instance-specific query.  Merge it instead of consuming another slot.
    return same_category_unbound[0][0]


def normalize_query_role(value: Any) -> str:
    role = sanitize_token_text(value)
    if role in ROLE_PRIORITY:
        return role
    return "context"


def merge_reason_text(*values: Any, max_chars: int = 240) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").strip().split())
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    merged = "; ".join(parts)
    if len(merged) <= max_chars:
        return merged
    return merged[: max(0, max_chars - 3)] + "..."


def sanitize_token_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def sanitize_hint_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9,; ]+", " ", text)
    return " ".join(text.split())


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
