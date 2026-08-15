import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/distribution_recalibration_results.json")
OUTPUT_PATH = Path("results/distribution_recalibration_comparison.png")


def main():
    # ------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------

    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    baseline = results["reliability"]["baseline"]
    recalibrated = results["reliability"]["distribution_recalibration"]

    # ------------------------------------------------------------
    # Extract reliability data
    # ------------------------------------------------------------

    baseline_confidence = baseline["bin_confidence"]
    baseline_accuracy = baseline["bin_accuracy"]
    baseline_count = baseline["bin_count"]

    recalibrated_confidence = recalibrated["bin_confidence"]
    recalibrated_accuracy = recalibrated["bin_accuracy"]
    recalibrated_count = recalibrated["bin_count"]

    # ------------------------------------------------------------
    # Remove empty bins
    # ------------------------------------------------------------

    baseline_points = [
        (confidence, accuracy, count)
        for confidence, accuracy, count in zip(
            baseline_confidence,
            baseline_accuracy,
            baseline_count,
        )
        if count > 0
    ]

    recalibrated_points = [
        (confidence, accuracy, count)
        for confidence, accuracy, count in zip(
            recalibrated_confidence,
            recalibrated_accuracy,
            recalibrated_count,
        )
        if count > 0
    ]

    # ------------------------------------------------------------
    # Create plot
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 7))

    # Perfect calibration
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    # Baseline
    if baseline_points:
        baseline_x = [point[0] for point in baseline_points]
        baseline_y = [point[1] for point in baseline_points]

        plt.plot(
            baseline_x,
            baseline_y,
            marker="o",
            label="Baseline",
        )

    # Distribution recalibration
    if recalibrated_points:
        recalibrated_x = [
            point[0] for point in recalibrated_points
        ]
        recalibrated_y = [
            point[1] for point in recalibrated_points
        ]

        plt.plot(
            recalibrated_x,
            recalibrated_y,
            marker="o",
            label="Distribution recalibration",
        )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")

    plt.title(
        "Reliability Diagram: Distribution Recalibration"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Distribution reliability diagram saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()