from __future__ import annotations

from dataclasses import replace

import pytest

from policy.roboharn_evo.agent.evidence_acquisition_policy import (
    EvidenceAcquisitionContext,
    EvidenceAcquisitionPolicy,
    EvidenceSituation,
    InformationAction,
)


def _context(**overrides: object) -> EvidenceAcquisitionContext:
    values: dict[str, object] = {
        "track_id": "track_0001",
        "arm": "right",
        "manipulation_phase": "grasp_candidate",
        "grasp_attempt_nonce": "attempt:1",
        "ee_pose": [
            0.1021,
            -0.1926,
            0.9305,
            0.384,
            0.596,
            0.379,
            -0.595,
        ],
        "camera_set": ["head", "third"],
        "capture_id": "capture:11",
        "skill_id": "skill_1",
        "physical_action_token": "physical:11",
        "viewpoint_token": "view:11",
    }
    values.update(overrides)
    return EvidenceAcquisitionContext(**values)  # type: ignore[arg-type]


def test_state_key_is_skill_independent_and_tolerates_pose_jitter() -> None:
    policy = EvidenceAcquisitionPolicy()
    first = _context()
    second = _context(
        skill_id="skill_after_replan",
        camera_set=["third", "HEAD", "head"],
        ee_pose=[
            0.1022,
            -0.1927,
            0.9306,
            -0.384,
            -0.596,
            -0.379,
            0.595,
        ],
    )

    assert policy.state_key(first) == policy.state_key(second)
    assert "skill_id" not in policy.state_key(first).as_dict()


def test_same_state_budget_survives_skill_and_situation_changes() -> None:
    policy = EvidenceAcquisitionPolicy()
    first = policy.decide(
        _context(skill_id="skill_1"),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )
    repeated = policy.decide(
        _context(
            skill_id="skill_2",
            capture_id="capture:12",
        ),
        situation=EvidenceSituation.TRANSIENT_BACKEND_RETRY,
    )

    assert first["allow_reobserve"] is True
    assert first["stationary_reobserves_used"] == 1
    assert repeated["allow_reobserve"] is False
    assert repeated["exhausted_reason"] == (
        "same_state_transient_backend_retry_budget_exhausted"
    )


def test_transient_backend_retry_is_bounded_to_one_same_state_retry() -> None:
    policy = EvidenceAcquisitionPolicy()

    first = policy.decide(
        _context(),
        situation=EvidenceSituation.TRANSIENT_BACKEND_RETRY,
    )
    second = policy.decide(
        _context(),
        situation=EvidenceSituation.TRANSIENT_BACKEND_RETRY,
    )

    assert first["allow_reobserve"] is True
    assert first["next_information_action"] == (
        InformationAction.RETRY_TRANSIENT_BACKEND.value
    )
    assert second["allow_reobserve"] is False
    assert second["next_information_action"] == (
        InformationAction.CHANGE_CAMERA_OR_ESCALATE_BACKEND_FAILURE.value
    )
    assert second["exhausted_reason"] == (
        "same_state_transient_backend_retry_budget_exhausted"
    )


def test_occlusion_requires_information_changing_motion_immediately() -> None:
    policy = EvidenceAcquisitionPolicy()

    decision = policy.decide(
        _context(),
        situation=EvidenceSituation.OCCLUSION,
    )

    assert decision["allow_reobserve"] is False
    assert decision["next_information_action"] == (
        InformationAction.CHANGE_VIEWPOINT_OR_CLEAR_OCCLUSION.value
    )
    assert decision["exhausted_reason"] == (
        "same_state_reobserve_cannot_clear_occlusion"
    )
    assert decision["stationary_reobserves_used"] == 0


@pytest.mark.parametrize(
    ("sufficient", "expected_action", "expected_reason"),
    [
        (
            True,
            InformationAction.ACCEPT_STRONG_GRASP_EVIDENCE.value,
            "",
        ),
        (
            False,
            InformationAction.ACQUIRE_INDEPENDENT_CAMERA_VIEW.value,
            "same_camera_reobserve_is_not_independent_evidence",
        ),
    ],
)
def test_strong_grasp_evidence_never_loops_in_same_view(
    sufficient: bool,
    expected_action: str,
    expected_reason: str,
) -> None:
    policy = EvidenceAcquisitionPolicy()

    decision = policy.decide(
        _context(),
        situation=EvidenceSituation.STRONG_GRASP_EVIDENCE,
        evidence_sufficient=sufficient,
    )

    assert decision["allow_reobserve"] is False
    assert decision["next_information_action"] == expected_action
    assert decision["exhausted_reason"] == expected_reason


def test_ambiguous_grasp_gets_one_observation_then_diagnostic_action() -> None:
    policy = EvidenceAcquisitionPolicy()

    first = policy.decide(
        _context(),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )
    exhausted = policy.decide(
        _context(capture_id="capture:12"),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )

    assert first["allow_reobserve"] is True
    assert first["next_information_action"] == (
        InformationAction.STATIONARY_REOBSERVE_ONCE.value
    )
    assert exhausted["allow_reobserve"] is False
    assert exhausted["next_information_action"] == (
        InformationAction.PERFORM_CONTROLLED_RETURN_AND_REGRASP.value
    )
    assert exhausted["exhausted_reason"] == (
        "same_state_ambiguous_grasp_reobserve_budget_exhausted"
    )


def test_proven_failed_grasp_requires_clearance_without_observing() -> None:
    decision = EvidenceAcquisitionPolicy().decide(
        _context(),
        situation=EvidenceSituation.PROVEN_FAILED_GRASP,
    )

    assert decision["allow_reobserve"] is False
    assert decision["next_information_action"] == (
        InformationAction.PERFORM_FAILED_GRASP_CLEARANCE.value
    )
    assert decision["stationary_reobserve_limit"] == 0


def test_release_stability_uses_two_distinct_captures_across_skills() -> None:
    policy = EvidenceAcquisitionPolicy()

    first = policy.decide(
        _context(
            manipulation_phase="release_pending_verification",
            grasp_attempt_nonce="release:1",
            capture_id="capture:20",
            skill_id="place_skill",
        ),
        situation=EvidenceSituation.RELEASE_STABILITY,
    )
    second = policy.decide(
        _context(
            manipulation_phase="release_pending_verification",
            grasp_attempt_nonce="release:1",
            capture_id="capture:21",
            skill_id="verification_skill",
        ),
        situation=EvidenceSituation.RELEASE_STABILITY,
    )

    assert first["allow_reobserve"] is True
    assert first["next_information_action"] == (
        InformationAction.ACQUIRE_DISTINCT_RELEASE_CAPTURE.value
    )
    assert first["distinct_capture_count"] == 1
    assert second["allow_reobserve"] is False
    assert second["next_information_action"] == (
        InformationAction.EVALUATE_RELEASE_STABILITY.value
    )
    assert second["exhausted_reason"] == ""
    assert second["distinct_capture_count"] == 2


def test_release_stability_rejects_duplicate_capture_without_looping() -> None:
    policy = EvidenceAcquisitionPolicy()
    context = _context(
        manipulation_phase="release_pending_verification",
        grasp_attempt_nonce="release:1",
        capture_id="capture:20",
    )

    assert policy.decide(
        context,
        situation=EvidenceSituation.RELEASE_STABILITY,
    )["allow_reobserve"] is True
    duplicate = policy.decide(
        replace(context, skill_id="new_skill"),
        situation=EvidenceSituation.RELEASE_STABILITY,
    )

    assert duplicate["allow_reobserve"] is False
    assert duplicate["next_information_action"] == (
        InformationAction.CHANGE_VIEWPOINT_OR_PHYSICAL_RECHECK.value
    )
    assert duplicate["exhausted_reason"] == (
        "release_reobserve_did_not_produce_distinct_capture"
    )


def test_identity_repair_keeps_two_distinct_capture_contract() -> None:
    policy = EvidenceAcquisitionPolicy()
    first_context = _context(
        manipulation_phase="relocation_pending",
        capture_id="capture:30",
    )

    first = policy.decide(
        first_context,
        situation=EvidenceSituation.TEMPORAL_IDENTITY_REPAIR,
    )
    second = policy.decide(
        replace(first_context, capture_id="capture:31"),
        situation=EvidenceSituation.TEMPORAL_IDENTITY_REPAIR,
    )

    assert first["allow_reobserve"] is True
    assert first["next_information_action"] == (
        InformationAction.ACQUIRE_DISTINCT_IDENTITY_CAPTURE.value
    )
    assert second["allow_reobserve"] is False
    assert second["next_information_action"] == (
        InformationAction.EVALUATE_IDENTITY_REPAIR.value
    )
    assert second["distinct_capture_count"] == 2


def test_physical_action_token_resets_even_at_same_quantized_pose() -> None:
    policy = EvidenceAcquisitionPolicy()
    context = _context(physical_action_token="physical:1")

    assert policy.decide(
        context,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is True
    assert policy.decide(
        context,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is False

    after_action = policy.decide(
        replace(context, physical_action_token="physical:2"),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )
    assert after_action["allow_reobserve"] is True


def test_scene_budget_status_is_read_only_and_resets_after_physical_action() -> None:
    policy = EvidenceAcquisitionPolicy()
    context = _context()

    before = policy.stationary_scene_budget_status(
        context,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )
    assert before["stationary_scene_reobserves_used"] == 0
    assert before["stationary_scene_reobserve_budget_exhausted"] is False

    assert policy.decide(
        context,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is True
    exhausted = policy.stationary_scene_budget_status(
        context,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )
    assert exhausted["stationary_scene_reobserves_used"] == 1
    assert exhausted["stationary_scene_reobserve_limit"] == 1
    assert exhausted["stationary_scene_reobserve_budget_exhausted"] is True

    after_action = policy.stationary_scene_budget_status(
        replace(context, physical_action_token="physical:12"),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )
    assert after_action["stationary_scene_reobserves_used"] == 0
    assert after_action["stationary_scene_reobserve_budget_exhausted"] is False


def test_only_physical_state_or_viewpoint_change_renews_scene_budget() -> None:
    policy = EvidenceAcquisitionPolicy()
    context = _context()

    def consume(current: EvidenceAcquisitionContext) -> None:
        assert policy.decide(
            current,
            situation=EvidenceSituation.AMBIGUOUS_GRASP,
        )["allow_reobserve"] is True
        assert policy.decide(
            current,
            situation=EvidenceSituation.AMBIGUOUS_GRASP,
        )["allow_reobserve"] is False

    consume(context)
    moved = replace(
        context,
        ee_pose=[
            0.1221,
            -0.1926,
            0.9305,
            0.384,
            0.596,
            0.379,
            -0.595,
        ],
    )
    assert policy.decide(
        moved,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is False
    after_physical_action = replace(
        moved,
        physical_action_token="physical:12",
    )
    consume(after_physical_action)
    added_camera = replace(
        after_physical_action,
        camera_set=["head", "third", "right"],
    )
    consume(added_camera)
    changed_view = replace(
        added_camera,
        viewpoint_token="view:12",
    )
    consume(changed_view)
    new_attempt = replace(
        changed_view,
        grasp_attempt_nonce="attempt:2",
    )
    assert policy.decide(
        new_attempt,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is False


def test_target_arm_and_skill_changes_do_not_renew_scene_budget() -> None:
    policy = EvidenceAcquisitionPolicy()
    context = _context()

    assert policy.decide(
        context,
        situation=EvidenceSituation.TRANSIENT_BACKEND_RETRY,
    )["allow_reobserve"] is True
    relabeled_same_scene = replace(
        context,
        track_id="track_0002",
        arm="left",
        manipulation_phase="scene_observation",
        grasp_attempt_nonce="",
        skill_id="skill_after_replan",
    )
    blocked = policy.decide(
        relabeled_same_scene,
        situation=EvidenceSituation.TRANSIENT_BACKEND_RETRY,
    )

    assert blocked["allow_reobserve"] is False
    assert blocked["exhausted_reason"] == (
        "same_physical_state_reobserve_budget_exhausted"
    )


def test_invalid_policy_configuration_and_missing_identity_fail_fast() -> None:
    with pytest.raises(ValueError):
        EvidenceAcquisitionPolicy(translation_quantum_m=0.0)
    with pytest.raises(ValueError):
        EvidenceAcquisitionPolicy(release_distinct_capture_limit=1)
    with pytest.raises(ValueError):
        EvidenceAcquisitionPolicy().state_key(
            _context(track_id="")
        )


def test_reset_is_explicit_episode_boundary_not_a_skill_transition() -> None:
    policy = EvidenceAcquisitionPolicy()
    context = _context(skill_id="skill_1")

    assert policy.decide(
        context,
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is True
    assert policy.decide(
        replace(context, skill_id="skill_2"),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is False

    policy.reset()

    assert policy.decide(
        replace(context, skill_id="skill_3"),
        situation=EvidenceSituation.AMBIGUOUS_GRASP,
    )["allow_reobserve"] is True
