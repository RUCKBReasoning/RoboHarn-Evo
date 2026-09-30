from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, get_type_hints

from roboharn_evo.agent.components.agent_tools.local_skill_registry import LocalSkillRegistry
from roboharn_evo.agent.components.memory_manager import MemoryManager
from roboharn_evo.agent.components.service_manager.service_types import ServiceRegister, ToolDef
from roboharn_evo.agent.paths import (
    eval_result_dir,
    resolve_writable_path,
    roboharn_skills_dir,
    writable_workspace_root,
)
from roboharn_evo.agent.perception import SAM2SegmentationClient, SAM3SegmentationClient

AgentToolsFunc = Callable[..., str | Awaitable[str]]


def extract_param_docs(func) -> dict[str, str]:
    doc = inspect.getdoc(func)
    if not doc:
        return {}
    param_docs: dict[str, str] = {}
    for line in doc.splitlines():
        line = line.strip()
        if ":" in line:
            name, desc = line.split(":", 1)
            param_docs[name.strip()] = desc.strip()
    return param_docs


def build_tool_from_func(func: AgentToolsFunc, service_name: str) -> ToolDef:
    name = func.__name__
    doc = inspect.getdoc(func) or "No description."
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    param_docs = extract_param_docs(func)
    required: list[str] = []
    properties: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        hint = type_hints.get(param_name, str)
        param_type = "string"
        if hint is bool:
            param_type = "boolean"
        elif hint is int:
            param_type = "integer"
        elif hint is float:
            param_type = "number"
        properties[param_name] = {"type": param_type, "description": param_docs.get(param_name, f"参数 {param_name}")}
    return ToolDef(
        service_name=service_name,
        tool_name=name,
        description=doc,
        parameters={"type": "object", "required": required, "properties": properties},
    )


class AgentTools:
    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        agent_card: Any,
        agent_instance: Optional[Any] = None,
    ) -> None:
        self._agent_card_ref = agent_card
        self._memory_manager_ref = memory_manager
        self._agent_instance = agent_instance
        self._service_name = "AgentTools"
        self._skill_service_name = "SkillTools"
        self._skill_registry = LocalSkillRegistry(
            configured_paths=[str(roboharn_skills_dir())],
            workspace_root=str(writable_workspace_root()),
        )
        self._agent_services_register = ServiceRegister(
            service_name=self._service_name,
            description="Internal RoboHarn-Evo agent tools.",
            is_activation=True,
            is_agent_service=True,
        )
        self._skill_services_register = ServiceRegister(
            service_name=self._skill_service_name,
            description=self._skill_registry.build_service_description(),
            is_activation=True,
            is_agent_service=True,
        )
        self._tool_registry: dict[str, dict[str, AgentToolsFunc]] = {}
        self._pending_context_injections: list[dict[str, Any]] = []

    async def init_agent_tools(self) -> None:
        await self.register_activate_service_tool()
        await self.register_skill_service_tool()

    async def register_activate_service_tool(self) -> None:
        tool_methods: list[AgentToolsFunc] = [
            self.fetch_env,
            self.segment_object,
            self.ensure_run_artifacts,
            self.append_jsonl_record,
        ]
        for method in tool_methods:
            tool = build_tool_from_func(method, service_name=self._service_name)
            self._agent_services_register.tools_list.append(tool)
            await self.register_tool(service_name=tool.service_name, tool_name=tool.tool_name, func=method)

    async def register_skill_service_tool(self) -> None:
        tool_methods: list[AgentToolsFunc] = [
            self.list_skills,
            self.get_skill_details,
        ]
        for method in tool_methods:
            tool = build_tool_from_func(method, service_name=self._skill_service_name)
            self._skill_services_register.tools_list.append(tool)
            await self.register_tool(service_name=tool.service_name, tool_name=tool.tool_name, func=method)

    async def fetch_env(self) -> str:
        """获取当前环境摘要以及最近一张图像上下文。"""
        if self._agent_instance is None:
            return json.dumps({"error": "agent instance unavailable"}, ensure_ascii=False)
        env_summary = self._agent_instance._get_env_summary() if hasattr(self._agent_instance, '_get_env_summary') else ''
        latest_snapshot = getattr(self._agent_instance, 'latest_snapshot', None)
        if latest_snapshot is not None:
            import base64
            from io import BytesIO
            from PIL import Image
            pil_image = Image.fromarray(latest_snapshot.head_rgb)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            self._pending_context_injections.append({"kind": "robot_image", "frame_id": latest_snapshot.step_count, "data_url": data_url})
        return json.dumps({"env_summary": env_summary}, ensure_ascii=False)

    async def segment_object(
        self,
        object_id: str,
        camera: str = "head",
        text_prompt: str = "",
        bbox_xyxy: str = "",
        point_coords: str = "",
        point_labels: str = "",
        backend: str = "sam3",
        service_url: str = "",
    ) -> str:
        """用 SAM2/SAM3 对当前相机图像中的目标做分割。
        object_id: 目标名。
        camera: head、left、right 或 third。
        text_prompt: SAM3 文本概念，例如 button 或 drawer handle；为空时默认使用 object_id。
        bbox_xyxy: SAM2 JSON 数组字符串 [x1, y1, x2, y2]。
        point_coords: SAM2 JSON 数组字符串 [[x, y], ...]。
        point_labels: SAM2 JSON 数组字符串，1 为前景点，0 为背景点。
        backend: sam3 或 sam2；默认 sam3 文本 grounding。
        service_url: 分割服务地址；默认按 backend 读取 ROBOHARN_EVO_SAM3_SERVICE_URL 或 ROBOHARN_EVO_SAM2_SERVICE_URL。
        """
        if self._agent_instance is None:
            return json.dumps({"success": False, "error": "agent instance unavailable"}, ensure_ascii=False)
        latest_snapshot = getattr(self._agent_instance, "latest_snapshot", None)
        if latest_snapshot is None:
            return json.dumps({"success": False, "error": "latest snapshot unavailable"}, ensure_ascii=False)

        image = self._select_snapshot_image(latest_snapshot, camera)
        if image is None:
            return json.dumps({"success": False, "error": f"unsupported camera: {camera}"}, ensure_ascii=False)

        import imageio.v3 as iio

        step = int(getattr(latest_snapshot, "step_count", 0))
        base_dir = resolve_writable_path(
            os.getenv(
                "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR",
                str(eval_result_dir("roboharn_segmentation")),
            )
        )
        image_dir = base_dir / "inputs"
        mask_dir = base_dir / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{camera}_step_{step:06d}.png"
        iio.imwrite(image_path, image)

        normalized_backend = str(backend or "").strip().lower()
        if normalized_backend not in {"sam2", "sam3"}:
            normalized_backend = "sam3"
        if normalized_backend == "sam2":
            result = self._segment_with_sam2(
                service_url=service_url,
                image_path=image_path,
                object_id=object_id,
                bbox_xyxy=bbox_xyxy,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_dir=mask_dir,
            )
        else:
            result = self._segment_with_sam3(
                service_url=service_url,
                image_path=image_path,
                object_id=object_id,
                text_prompt=text_prompt,
                mask_dir=mask_dir,
            )
        if not isinstance(result, dict):
            return json.dumps({"success": False, "error": "segmentation backend returned invalid result"}, ensure_ascii=False)
        if not result.get("success", False):
            return json.dumps(result, ensure_ascii=False)

        robot_state = {}
        if hasattr(self._agent_instance, "_get_robot_state"):
            robot_state = self._agent_instance._get_robot_state()
        result.update(
            {
                "backend": normalized_backend,
                "camera": camera,
                "env_step": step,
                "robot_state": robot_state,
            }
        )
        return json.dumps(result, ensure_ascii=False)

    def _segment_with_sam2(
        self,
        *,
        service_url: str,
        image_path: Path,
        object_id: str,
        bbox_xyxy: str,
        point_coords: str,
        point_labels: str,
        mask_dir: Path,
    ) -> dict[str, Any]:
        bbox = self._parse_optional_json_array(bbox_xyxy, "bbox_xyxy")
        points = self._parse_optional_json_array(point_coords, "point_coords")
        labels = self._parse_optional_json_array(point_labels, "point_labels")
        if bbox is None and points is None:
            return {"success": False, "error": "sam2 requires bbox_xyxy or point_coords", "image_path": str(image_path)}

        client = SAM2SegmentationClient(
            base_url=service_url or os.getenv("ROBOHARN_EVO_SAM2_SERVICE_URL", "http://127.0.0.1:9201"),
            timeout_sec=float(os.getenv("ROBOHARN_EVO_SAM2_TIMEOUT_SEC", "30")),
        )
        try:
            return client.segment_image(
                image_path=image_path,
                object_id=object_id,
                bbox_xyxy=bbox,
                point_coords=points,
                point_labels=labels,
                output_dir=mask_dir,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "image_path": str(image_path)}

    def _segment_with_sam3(
        self,
        *,
        service_url: str,
        image_path: Path,
        object_id: str,
        text_prompt: str,
        mask_dir: Path,
    ) -> dict[str, Any]:
        prompt = str(text_prompt or object_id).strip()
        if not prompt:
            return {"success": False, "error": "sam3 requires text_prompt or object_id", "image_path": str(image_path)}
        client = SAM3SegmentationClient(
            base_url=service_url or os.getenv("ROBOHARN_EVO_SAM3_SERVICE_URL", "http://127.0.0.1:9301"),
            timeout_sec=float(os.getenv("ROBOHARN_EVO_SAM3_TIMEOUT_SEC", "60")),
        )
        try:
            threshold_text = os.getenv("ROBOHARN_EVO_SAM3_CONFIDENCE_THRESHOLD", "").strip()
            return client.segment_image(
                image_path=image_path,
                object_id=object_id,
                text_prompt=prompt,
                output_dir=mask_dir,
                top_k=int(os.getenv("ROBOHARN_EVO_SAM3_TOP_K", "3")),
                confidence_threshold=float(threshold_text) if threshold_text else None,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "image_path": str(image_path)}

    def _select_snapshot_image(self, snapshot: Any, camera: str):
        normalized = str(camera or "head").strip().lower().replace("_camera", "")
        if normalized == "head":
            return getattr(snapshot, "head_rgb", None)
        if normalized == "left":
            return getattr(snapshot, "left_rgb", None)
        if normalized == "right":
            return getattr(snapshot, "right_rgb", None)
        if normalized == "third":
            return getattr(snapshot, "third_rgb", None)
        return None

    def _parse_optional_json_array(self, value: str, name: str):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"{name} must be a JSON array")
        return parsed

    async def flush_pending_context_injections(self) -> None:
        while self._pending_context_injections:
            injection = self._pending_context_injections.pop(0)
            if injection.get("kind") == "robot_image":
                self._memory_manager_ref.add_robot_image(
                    image_data_url=str(injection["data_url"]),
                    frame_id=int(injection["frame_id"]),
                )
                if self._memory_manager_ref.runtime_tree.current_task_node and len(self._memory_manager_ref.runtime_tree.current_task_node.contexts) > 50:
                    self._memory_manager_ref.compress_current_memory(drop_n=1)

    async def ensure_run_artifacts(self, run_dir: str) -> str:
        """创建 workflow 运行目录。:param run_dir: 运行目录"""
        base_dir = resolve_writable_path(run_dir)
        logs_dir = base_dir / 'logs'
        dataset_dir = base_dir / 'dataset'
        logs_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return json.dumps({"run_dir": str(base_dir), "logs_dir": str(logs_dir), "dataset_dir": str(dataset_dir), "created": True}, ensure_ascii=False)

    async def append_jsonl_record(self, file_path: str, record_json: str) -> str:
        """追加一条 JSONL 记录。:param file_path: JSONL路径 :param record_json: JSON字符串"""
        resolved = resolve_writable_path(file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        record = json.loads(record_json)
        serialized = json.dumps(record, ensure_ascii=False)
        with resolved.open('a', encoding='utf-8') as handle:
            handle.write(serialized + '\n')
        return json.dumps({"file_path": str(resolved), "appended": True, "record_preview": serialized[:200]}, ensure_ascii=False)

    async def list_skills(self) -> str:
        """列出当前可用的本地 skill。"""
        skills = [skill.summary_dict() for skill in self._skill_registry.list_skills(refresh=True)]
        return json.dumps({"skills": skills, "count": len(skills)}, ensure_ascii=False)

    async def get_skill_details(self, skill_name: str) -> str:
        """查看某个 skill 的完整说明。:param skill_name: skill 名"""
        skill = self._skill_registry.get_skill(skill_name, refresh=True)
        if skill is None:
            return json.dumps({"error": f"未找到 skill: {skill_name}"}, ensure_ascii=False)
        return json.dumps(skill.detail_dict(), ensure_ascii=False)

    async def register_tool(self, *, service_name: str, tool_name: str, func: AgentToolsFunc) -> None:
        self._tool_registry.setdefault(service_name, {})[tool_name] = func

    def get_tool(self, service_name: str, tool_name: str) -> Optional[AgentToolsFunc]:
        return self._tool_registry.get(service_name, {}).get(tool_name)

    async def tools_routing(self, service_name: str, tool_name: str, tool_args: dict[str, Any]) -> str:
        tool_func = self.get_tool(service_name, tool_name)
        if tool_func is None:
            raise ValueError(f"未找到工具函数: service={service_name}, tool={tool_name}")
        if inspect.iscoroutinefunction(tool_func):
            result = await tool_func(**tool_args)
        else:
            loop: AbstractEventLoop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: tool_func(**tool_args))
        await self.flush_pending_context_injections()
        return str(result)

    @property
    def service_registers(self) -> list[ServiceRegister]:
        return [self._agent_services_register, self._skill_services_register]
