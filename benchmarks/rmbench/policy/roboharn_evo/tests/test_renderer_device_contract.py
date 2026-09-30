from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "envs" / "renderer_device_contract.py"
)
SPEC = importlib.util.spec_from_file_location("renderer_device_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer_contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer_contract
SPEC.loader.exec_module(renderer_contract)


class FakeDevice:
    def __init__(
        self,
        alias: str,
        *,
        cuda_id: int = 0,
        pci_string: str = "0000:4d:00.0",
        is_cuda: bool = True,
        can_render: bool = True,
    ) -> None:
        self.alias = alias
        self.cuda_id = cuda_id
        self.pci_string = pci_string
        self.name = "Fake RTX"
        self._is_cuda = is_cuda
        self._can_render = can_render

    def __str__(self) -> str:
        return self.alias

    def is_cuda(self) -> bool:
        return self._is_cuda

    def can_render(self) -> bool:
        return self._can_render


class FakeRenderSystem:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device


class FakeScene:
    def __init__(self, systems: list[object]) -> None:
        self.systems = systems
        self.render_system = next(
            system for system in systems if isinstance(system, FakeRenderSystem)
        )


class FakePhysxSystem:
    pass


class FakeRenderer:
    def __init__(self, device: FakeDevice | None = None) -> None:
        self.device = device


class FakeRenderApi:
    def __init__(self, owner: "FakeSapien") -> None:
        self.owner = owner

    def SapienRenderer(self, device: FakeDevice) -> FakeRenderer:
        self.owner.bottom_renderer_devices.append(device)
        return FakeRenderer(device)

    def RenderSystem(self, device: FakeDevice) -> FakeRenderSystem:
        self.owner.render_system_devices.append(device)
        return FakeRenderSystem(device)


class FakePhysxApi:
    def __init__(self, owner: "FakeSapien") -> None:
        self.owner = owner

    def set_scene_config(self, config: object) -> None:
        self.owner.scene_configs.append(config)

    def PhysxCpuSystem(self) -> FakePhysxSystem:
        return FakePhysxSystem()


class FakeSapien:
    def __init__(self, *, selected_device: FakeDevice | None = None) -> None:
        self.selected_device = selected_device
        self.device_aliases: list[str] = []
        self.top_renderer_calls = 0
        self.bottom_renderer_devices: list[FakeDevice] = []
        self.render_system_devices: list[FakeDevice] = []
        self.scene_configs: list[object] = []
        self.scene_systems: list[list[object]] = []
        self.render = FakeRenderApi(self)
        self.physx = FakePhysxApi(self)

    def Device(self, alias: str) -> FakeDevice:
        self.device_aliases.append(alias)
        if self.selected_device is not None:
            return self.selected_device
        return FakeDevice(alias)

    def SapienRenderer(self) -> FakeRenderer:
        self.top_renderer_calls += 1
        return FakeRenderer()

    def Scene(self, systems: list[object]) -> FakeScene:
        self.scene_systems.append(systems)
        return FakeScene(systems)


class FakeEngine:
    def __init__(self) -> None:
        self.renderers: list[FakeRenderer] = []
        self.legacy_scene_configs: list[object] = []
        self.legacy_scene = object()

    def set_renderer(self, renderer: FakeRenderer) -> None:
        self.renderers.append(renderer)

    def create_scene(self, config: object) -> object:
        self.legacy_scene_configs.append(config)
        return self.legacy_scene


class RendererDeviceContractTest(unittest.TestCase):
    def test_ray_tracing_denoiser_defaults_to_oidn_and_accepts_optix(self) -> None:
        self.assertEqual(
            renderer_contract.configured_ray_tracing_denoiser({}),
            "oidn",
        )
        self.assertEqual(
            renderer_contract.configured_ray_tracing_denoiser(
                {"RMBENCH_RAY_TRACING_DENOISER": " OptiX "}
            ),
            "optix",
        )
        self.assertEqual(
            renderer_contract.configured_ray_tracing_denoiser(
                {"RMBENCH_RAY_TRACING_DENOISER": "none"}
            ),
            "none",
        )
        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "RMBENCH_RAY_TRACING_DENOISER",
        ):
            renderer_contract.configured_ray_tracing_denoiser(
                {"RMBENCH_RAY_TRACING_DENOISER": "invalid"}
            )

    def test_unset_contract_preserves_exact_legacy_construction(self) -> None:
        sapien = FakeSapien()
        engine = FakeEngine()
        config = object()

        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ={}
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, config, binding
        )
        provenance = renderer_contract.validate_renderer_binding(binding, scene)

        self.assertEqual(binding.mode, "legacy_auto")
        self.assertEqual(sapien.top_renderer_calls, 1)
        self.assertEqual(sapien.bottom_renderer_devices, [])
        self.assertIs(scene, engine.legacy_scene)
        self.assertEqual(engine.legacy_scene_configs, [config])
        self.assertEqual(sapien.scene_configs, [])
        self.assertTrue(provenance["validation_passed"])

    def test_explicit_pci_binding_reuses_exact_device_for_renderer_and_scene(self) -> None:
        selected = FakeDevice("pci:0000:4d:00.0", pci_string="0000:4D:00.0")
        sapien = FakeSapien(selected_device=selected)
        engine = FakeEngine()
        config = object()
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:4d:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "00000000:4d:00.0",
            "RMBENCH_EXPECTED_PHYSICAL_GPU": "4",
            "CUDA_VISIBLE_DEVICES": "GPU-abcd",
        }

        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, config, binding
        )
        provenance = renderer_contract.validate_renderer_binding(binding, scene)

        self.assertEqual(sapien.device_aliases, ["pci:0000:4d:00.0"])
        self.assertEqual(sapien.top_renderer_calls, 0)
        self.assertEqual(sapien.bottom_renderer_devices, [selected])
        self.assertEqual(sapien.render_system_devices, [selected])
        self.assertEqual(sapien.scene_configs, [config])
        self.assertEqual(engine.legacy_scene_configs, [])
        self.assertIs(scene.render_system.device, selected)
        self.assertTrue(provenance["strict"])
        self.assertTrue(provenance["validation_passed"])
        self.assertEqual(provenance["requested_device"], "pci:0000:4d:00.0")
        self.assertEqual(provenance["selected_device_info"]["cuda_id"], 0)
        self.assertEqual(
            provenance["selected_device_info"]["pci_bus_id_normalized"],
            "0000:4d:00.0",
        )
        self.assertEqual(provenance["expected_physical_gpu"], "4")

    def test_installed_sapien_api_exposes_required_explicit_device_hooks(self) -> None:
        try:
            import sapien.core as installed_sapien
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"installed SAPIEN cannot be imported: {exc}")

        self.assertTrue(callable(installed_sapien.Device))
        self.assertTrue(callable(installed_sapien.render.SapienRenderer))
        self.assertTrue(callable(installed_sapien.render.RenderSystem))
        self.assertTrue(callable(installed_sapien.physx.PhysxCpuSystem))
        self.assertTrue(callable(installed_sapien.physx.set_scene_config))

    def test_viewer_accepts_the_device_bound_lower_level_renderer(self) -> None:
        try:
            from sapien.utils.viewer import Viewer
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"installed SAPIEN viewer cannot be imported: {exc}")

        renderer = type("Renderer", (), {"_internal_context": object()})()
        fake_window = mock.Mock()
        fake_window.should_close = False
        with mock.patch(
            "sapien.utils.viewer.viewer.RenderWindow", return_value=fake_window
        ), mock.patch(
            "sapien.utils.viewer.viewer.sapien.render.get_imgui_ini_filename",
            return_value="/tmp/rmbench_renderer_contract_imgui.ini",
        ):
            viewer = Viewer(renderer=renderer, plugins=[])

        self.assertIs(viewer.renderer_context, renderer._internal_context)

    def test_strict_binding_rejects_physical_cuda_index_inside_child(self) -> None:
        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "requires RMBENCH_RENDER_DEVICE=cuda:0",
        ):
            renderer_contract.create_renderer_binding(
                FakeSapien(),
                FakeEngine(),
                environ={
                    "RMBENCH_RENDER_DEVICE": "cuda:4",
                    "RMBENCH_RENDER_DEVICE_STRICT": "1",
                    "CUDA_VISIBLE_DEVICES": "4",
                },
            )

    def test_strict_pci_binding_rejects_multiple_visible_cuda_devices(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:4d:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "true",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "0000:4d:00.0",
            "CUDA_VISIBLE_DEVICES": "GPU-abcd,GPU-ef01",
        }
        sapien = FakeSapien()
        engine = FakeEngine()
        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, object(), binding
        )

        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "exactly one CUDA_VISIBLE_DEVICES entry",
        ):
            renderer_contract.validate_renderer_binding(binding, scene)

    def test_strict_pci_binding_requires_uuid_visible_device_token(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:4d:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "0000:4d:00.0",
            "CUDA_VISIBLE_DEVICES": "4",
        }
        sapien = FakeSapien(
            selected_device=FakeDevice(
                "pci:0000:4d:00.0", pci_string="0000:4d:00.0"
            )
        )
        engine = FakeEngine()
        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, object(), binding
        )

        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "UUID",
        ):
            renderer_contract.validate_renderer_binding(binding, scene)

    def test_strict_pci_binding_rejects_requested_alias_mismatch(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:5e:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "0000:4d:00.0",
            "CUDA_VISIBLE_DEVICES": "GPU-abcd",
        }
        sapien = FakeSapien(
            selected_device=FakeDevice(
                "pci:0000:4d:00.0", pci_string="0000:4d:00.0"
            )
        )
        engine = FakeEngine()
        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "alias must exactly match.*expected PCI",
        ):
            renderer_contract.create_renderer_binding(
                sapien, engine, environ=environment
            )

    def test_strict_pci_binding_rejects_nonzero_logical_cuda_id(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:4d:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "0000:4d:00.0",
            "CUDA_VISIBLE_DEVICES": "GPU-abcd",
        }
        sapien = FakeSapien(
            selected_device=FakeDevice(
                "pci:0000:4d:00.0",
                cuda_id=4,
                pci_string="0000:4d:00.0",
            )
        )
        engine = FakeEngine()
        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, object(), binding
        )

        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "logical CUDA ID 4.*expected 0",
        ):
            renderer_contract.validate_renderer_binding(binding, scene)

    def test_strict_binding_rejects_pci_mismatch(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:5e:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "00000000:5e:00.0",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "CUDA_VISIBLE_DEVICES": "GPU-abcd",
        }
        sapien = FakeSapien(
            selected_device=FakeDevice(
                "pci:0000:5e:00.0", pci_string="0000:4d:00.0"
            )
        )
        engine = FakeEngine()
        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, object(), binding
        )

        with self.assertRaisesRegex(
            renderer_contract.RendererDeviceContractError,
            "does not match expected",
        ):
            renderer_contract.validate_renderer_binding(binding, scene)

    def test_strict_validation_error_carries_auditable_failure_provenance(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:5e:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "00000000:5e:00.0",
            "RMBENCH_EXPECTED_PHYSICAL_GPU": "4",
            "CUDA_VISIBLE_DEVICES": "GPU-abcd",
        }
        sapien = FakeSapien(
            selected_device=FakeDevice(
                "pci:0000:5e:00.0", pci_string="0000:4d:00.0"
            )
        )
        engine = FakeEngine()
        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, object(), binding
        )

        with self.assertRaises(
            renderer_contract.RendererDeviceContractError
        ) as raised:
            renderer_contract.validate_renderer_binding(binding, scene)

        error = raised.exception
        self.assertEqual(error.stage, "device_validation")
        self.assertIsNotNone(error.provenance)
        self.assertFalse(error.provenance["validation_passed"])
        self.assertEqual(error.provenance["expected_physical_gpu"], "4")
        self.assertEqual(error.provenance["expected_pci_bus_id"], "0000:5e:00.0")
        self.assertEqual(
            error.provenance["selected_device_info"]["pci_bus_id_normalized"],
            "0000:4d:00.0",
        )

        failure = renderer_contract.renderer_contract_failure_provenance(error)
        self.assertEqual(failure["failure_type"], "RendererDeviceContractError")
        self.assertEqual(failure["failure_stage"], "device_validation")
        self.assertFalse(failure["validation_passed"])
        self.assertIn("does not match expected", failure["validation_errors"][0])

    def test_nonstrict_binding_records_mismatch_without_raising(self) -> None:
        environment = {
            "RMBENCH_RENDER_DEVICE": "cuda:0",
            "RMBENCH_RENDER_DEVICE_STRICT": "0",
            "CUDA_VISIBLE_DEVICES": "4,5",
        }
        sapien = FakeSapien()
        engine = FakeEngine()
        binding = renderer_contract.create_renderer_binding(
            sapien, engine, environ=environment
        )
        scene = renderer_contract.create_scene_for_binding(
            sapien, engine, object(), binding
        )

        provenance = renderer_contract.validate_renderer_binding(binding, scene)

        self.assertFalse(provenance["validation_passed"])
        self.assertIn("exactly one", provenance["validation_errors"][0])

    def test_optional_provenance_file_is_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="renderer_binding_test_") as directory:
            path = Path(directory) / "eval_result" / "renderer_device_provenance.json"
            provenance = {"schema": "test", "validation_passed": True}

            with mock.patch.object(
                renderer_contract, "_tcm_project_root", return_value=Path(directory)
            ):
                result = renderer_contract.write_renderer_device_provenance(
                    provenance,
                    environ={
                        "RMBENCH_OUTPUT_ROOT": str(path.parent),
                        "ROBOHARN_EVO_OUTPUT_ROOT": str(path.parent),
                        "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(path),
                    },
                )

            self.assertEqual(result, path.resolve())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), provenance)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failure_provenance_atomically_replaces_existing_record(self) -> None:
        error = renderer_contract.RendererDeviceContractError(
            "strict renderer mismatch",
            stage="device_validation",
            provenance={
                "schema": "rmbench/renderer_device_binding/v1",
                "mode": "explicit",
                "strict": True,
                "validation_passed": False,
                "validation_errors": ["strict renderer mismatch"],
            },
        )
        failure = renderer_contract.renderer_contract_failure_provenance(error)

        with tempfile.TemporaryDirectory(prefix="renderer_failure_test_") as directory:
            path = Path(directory) / "eval_result" / "renderer_device_provenance.json"
            path.parent.mkdir(parents=True)
            original = {"schema": "old", "validation_passed": True}
            path.write_text(json.dumps(original) + "\n", encoding="utf-8")
            real_replace = renderer_contract.os.replace
            replacement_observations: list[tuple[Path, Path]] = []

            def assert_atomic_replace(source: object, destination: object) -> None:
                source_path = Path(source)
                destination_path = Path(destination)
                self.assertTrue(source_path.is_file())
                self.assertEqual(source_path.parent, destination_path.parent)
                self.assertEqual(
                    json.loads(destination_path.read_text(encoding="utf-8")),
                    original,
                )
                replacement_observations.append((source_path, destination_path))
                real_replace(source, destination)

            with mock.patch.object(
                renderer_contract.os,
                "replace",
                side_effect=assert_atomic_replace,
            ), mock.patch.object(
                renderer_contract, "_tcm_project_root", return_value=Path(directory)
            ):
                result = renderer_contract.write_renderer_device_provenance(
                    failure,
                    environ={
                        "RMBENCH_OUTPUT_ROOT": str(path.parent),
                        "ROBOHARN_EVO_OUTPUT_ROOT": str(path.parent),
                        "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(path)
                    },
                )

            self.assertEqual(result, path.resolve())
            self.assertEqual(len(replacement_observations), 1)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), failure)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
