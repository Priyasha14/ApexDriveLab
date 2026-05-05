import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetry.analysis import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two telemetry CSV files.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline = summarize(args.baseline)
    candidate = summarize(args.candidate)

    print(f"Baseline: {args.baseline}")
    print(f"Candidate: {args.candidate}")
    for key in ["duration_s", "avg_speed_kmh", "max_speed_kmh", "grip_saturation_pct", "off_track_pct"]:
        base_value = float(baseline[key])
        candidate_value = float(candidate[key])
        print(f"{key}: baseline={base_value:.3f}, candidate={candidate_value:.3f}, delta={candidate_value - base_value:.3f}")


if __name__ == "__main__":
    main()
