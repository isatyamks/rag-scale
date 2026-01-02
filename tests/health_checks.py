import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from tests.health import HealthCheckRunner


def main():
    settings = get_settings()
    runner = HealthCheckRunner(settings)
    success = runner.run_all_checks(verbose=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
