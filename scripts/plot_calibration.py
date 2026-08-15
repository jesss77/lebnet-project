import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    results_path = Path("results/temperature_scaling_results.json")

    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find {results_path}. "
            "Run train_recalibrator first."
        )

    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    baseline = results["reliability"]["baseline"]
    calibrated = results["reliability"]["temperature_scaled"]

    baseline_confidence = baseline["bin_confidence"]
    baseline_accuracy = baseline["bin_accuracy"]

    calibrated_confidence = calibrated["bin_confidence"]
    calibrated_accuracy = calibrated["bin_accuracy"]

    plt.figure(figsize=(8, 6))

    plt.plot(
        baseline_confidence,
        baseline_accuracy,
        marker="o",
        label="Baseline",
    )

    plt.plot(
        calibrated_confidence,
        calibrated_accuracy,
        marker="o",
        label="Temperature Scaling",
    )

    plt.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram: Baseline vs Temperature Scaling")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = Path("results/reliability_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Reliability diagram saved to: {output_path}")


if __name__ == "__main__":
    main()