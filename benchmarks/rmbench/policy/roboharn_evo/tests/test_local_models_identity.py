from __future__ import annotations

import unittest

from policy.roboharn_evo.scripts.local_models_identity import (
    ModelsIdentityError,
    models_identity_sha256,
)


def _payload(*, created: int, permission_id: str, max_model_len: int = 65536):
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3.5-397B-A17B-FP8",
                "object": "model",
                "created": created,
                "owned_by": "vllm",
                "root": "/models/Qwen3.5-397B-A17B-FP8",
                "parent": None,
                "max_model_len": max_model_len,
                "permission": [
                    {
                        "id": permission_id,
                        "object": "model_permission",
                        "created": created,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            }
        ],
    }


class LocalModelsIdentityTest(unittest.TestCase):
    def test_ignores_dynamic_timestamps_and_permission_ids(self) -> None:
        first = _payload(created=1, permission_id="modelperm-first")
        second = _payload(created=2, permission_id="modelperm-second")
        self.assertEqual(models_identity_sha256(first), models_identity_sha256(second))

    def test_changes_when_stable_model_configuration_changes(self) -> None:
        first = _payload(created=1, permission_id="modelperm-first")
        second = _payload(
            created=2,
            permission_id="modelperm-second",
            max_model_len=32768,
        )
        self.assertNotEqual(models_identity_sha256(first), models_identity_sha256(second))

    def test_rejects_model_card_without_id(self) -> None:
        payload = _payload(created=1, permission_id="modelperm-first")
        del payload["data"][0]["id"]
        with self.assertRaises(ModelsIdentityError):
            models_identity_sha256(payload)


if __name__ == "__main__":
    unittest.main()
