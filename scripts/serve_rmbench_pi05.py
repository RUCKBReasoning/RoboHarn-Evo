from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboharn_evo.services.pi05_executor import main  # noqa: E402

if __name__ == "__main__":
    main()
