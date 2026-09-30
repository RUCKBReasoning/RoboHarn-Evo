"""Explicit VLA-free backend for the shared Agent's pure tool-control mode."""


class ToolOnlyExecutor:
    def reset(self) -> None:
        pass

    def predict_action_chunk(self, **kwargs):
        raise RuntimeError("tool_only executor cannot emit VLA actions; enable Agent pure_tool_control")
