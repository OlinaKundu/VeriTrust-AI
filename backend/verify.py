import sys
from pathlib import Path

# Add tests directory
sys.path.insert(0, str(Path(__file__).parent))

from tests.verify_pipeline import run_tests

if __name__ == "__main__":
    run_tests()
