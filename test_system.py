from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tests.health_checks import main

if __name__ == "__main__":
    print("\nRunning pre-flight health checks...\n")
    sys.exit(main())
