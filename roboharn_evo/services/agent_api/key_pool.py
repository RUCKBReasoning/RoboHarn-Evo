"""Per-credential concurrency behind one existing Agent API endpoint."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Callable


@dataclass
class KeySlot:
    name: str
    client: Any = field(repr=False)
    limit: int
    active: int = 0


class ApiKeyPool:
    """The handler's global semaphore bounds admission; this assigns a key.

    A lease lasts for the entire buffered request, including stream consumption
    and retries. All workers using these credentials must share this pool.
    """

    def __init__(self, slots: list[KeySlot]):
        if not slots or any(type(slot.limit) is not int or slot.limit < 1 for slot in slots):
            raise ValueError("API key pool needs positive per-key concurrency limits")
        if len({slot.name for slot in slots}) != len(slots):
            raise ValueError("API key pool names must be unique")
        self.slots = slots
        self.capacity = sum(slot.limit for slot in slots)
        self._lock = threading.Lock()
        self._next = 0

    def acquire(self) -> KeySlot:
        with self._lock:
            for offset in range(len(self.slots)):
                index = (self._next + offset) % len(self.slots)
                slot = self.slots[index]
                if slot.active < slot.limit:
                    slot.active += 1
                    self._next = (index + 1) % len(self.slots)
                    return slot
        raise RuntimeError("API key pool is full; gateway admission must not exceed pool capacity")

    def release(self, slot: KeySlot) -> None:
        with self._lock:
            slot.active -= 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"capacity": self.capacity, "keys": [
                {"name": slot.name, "max_concurrent_requests": slot.limit,
                 "active_requests": slot.active}
                for slot in self.slots
            ]}

    def close(self) -> None:
        for slot in self.slots:
            slot.client.close()


def load_api_key_pool(path: str, *, client_factory: Callable[[str], Any]) -> ApiKeyPool:
    """Read secrets locally once; never export key values, paths or fingerprints."""
    config = Path(path).expanduser().resolve()
    data = json.loads(config.read_text())
    entries = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("API key pool config requires a non-empty keys list")
    credentials = []
    names: set[str] = set()
    for entry in entries:
        name = str(entry["name"]).strip()
        limit = entry["max_concurrent_requests"]
        if not name or name in names or type(limit) is not int or limit < 1:
            raise ValueError("API key pool requires unique names and positive integer limits")
        source = Path(entry["api_key_file"]).expanduser()
        if not source.is_absolute():
            source = config.parent / source
        try:
            payload = json.loads(source.read_text())
        except (OSError, ValueError):
            raise ValueError(f"API key file for pool member {name!r} could not be read") from None
        key = payload.get(entry.get("api_key_json_key", "OPENAI_API_KEY")) if isinstance(payload, dict) else None
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"API key file for pool member {name!r} has no non-empty key")
        key = key.strip()
        if any(key == existing[2] for existing in credentials):
            raise ValueError("Duplicate API credentials cannot provide independent pool slots")
        names.add(name)
        credentials.append((name, limit, key))
    slots = []
    try:
        for name, limit, key in credentials:
            slots.append(KeySlot(name, client_factory(key), limit))
    except BaseException:
        for slot in slots:
            slot.client.close()
        raise
    return ApiKeyPool(slots)
