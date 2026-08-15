import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():

    results_path = Path(
        "results/"
        "distribution_recalibration_results.json"
    )

    if not results_path.exists():

        raise FileNotFoundError(
            f"Could not find {results_path}. "
            "Run train_distribution_recalibrator first."
        )

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    baseline = results["baseline"]

    plt.figure(
        figsize=(9, 7)
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_reliability = (
        baseline["reliability"]
    )

    baseline_confidence = (
        baseline_reliability[
            "bin_confidence"
        ]
    )

    baseline_accuracy = (
        baseline_reliability[
            "bin_accuracy"
        ]
    )

    plt.plot(
        baseline_confidence,
        baseline_accuracy,
        marker="o",
        label="Baseline",
    )

    # --------------------------------------------------------
    # Distribution recalibrators
    # --------------------------------------------------------

    for (
        experiment_name,
        experiment,
    ) in results["experiments"].items():

        reliability = experiment[
            "reliability"
        ][
            "distribution_recalibration"
        ]

        confidence = reliability[
            "bin_confidence"
        ]

        accuracy = reliability[
            "bin_accuracy"
        ]

        plt.plot(
            confidence,
            accuracy,
            marker="o",
            label=experiment_name,
        )

    # --------------------------------------------------------
    # Perfect calibration
    # --------------------------------------------------------

    plt.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel(
        "Confidence"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Distribution Recalibration "
        "Capacity Comparison"
    )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    output_path = Path(
        "results/"
        "distribution_recalibration_comparison.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Distribution reliability diagram "
        f"saved to: {output_path}"
    )


if __name__ == "__main__":
    main()