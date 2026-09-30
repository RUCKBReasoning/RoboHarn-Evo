from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.models.robobrain_adapter import (
    DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
    RoboBrainPlannerAdapter,
    RoboBrainPlannerConfig,
    decode_remote_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve RoboBrain planner VLM over HTTP for RoboHarn-Evo deployment.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model-id", type=str, default="policy/roboharn_evo/checkpoints/robobrain/RoboBrain2.5-8B-NV")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    return parser.parse_args()


class RoboBrainPlannerHandler(BaseHTTPRequestHandler):
    adapter: RoboBrainPlannerAdapter | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/plan":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        if self.adapter is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "planner adapter is not initialized")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))
        request_payload = decode_remote_request(payload)
        prediction = self.adapter.predict_planner_step(
            task=request_payload["task"],
            previous_memory_text=request_payload["previous_memory_text"],
            planner_start_image=request_payload["planner_start_image"],
            planner_end_image=request_payload["planner_end_image"],
            planner_state=request_payload["planner_state"],
        )
        self._send_json(prediction)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    RoboBrainPlannerHandler.adapter = RoboBrainPlannerAdapter(
        RoboBrainPlannerConfig(
            mode="local",
            model_id=args.model_id,
            device_map=args.device_map,
            server_url="",
            timeout_sec=args.timeout_sec,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            prompt_template=DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
        )
    )
    server = ThreadingHTTPServer((args.host, args.port), RoboBrainPlannerHandler)
    print(f"[robobrain-planner] listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
