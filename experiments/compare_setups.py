import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.optimizer import format_result, random_search


def main() -> None:
    for setup in ["balanced", "high_downforce", "low_drag", "front_aero", "rear_aero"]:
        result = random_search(iterations=6, seed=11, setup_name=setup)
        print(f"{setup}: {format_result(result)}")


if __name__ == "__main__":
    main()
