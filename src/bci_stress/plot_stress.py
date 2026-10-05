"""Plot completed frozen stress summaries without loading signals or estimators."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

DECODERS = {
    "csp_lda": ("CSP + LDA", "#126A9C"),
    "bandpower_lda": ("Channel bandpower + LDA", "#C35B20"),
}
FAMILIES = {
    "gaussian": ("noise_levels", "Gaussian noise", "Noise standard deviation / training RMS"),
    "flatline": ("dropout_counts", "Channel flatlining", "Number of failed channels"),
}
METRICS = ("balanced_accuracy", "delta_balanced_accuracy")


def _load_results(report_dir: Path) -> tuple[list[dict], dict, dict]:
    status = json.loads((report_dir / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Only a completed frozen experiment may be plotted")
    config = json.loads((report_dir / "stress_config.json").read_text())
    baseline = json.loads((report_dir / "clean_baseline_metrics.json").read_text())
    with (report_dir / "summary.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {
        (family, float(severity), decoder)
        for family, (grid, _, _) in FAMILIES.items()
        for severity in config[grid]
        for decoder in DECODERS
    }
    observed = set()
    for row in rows:
        key = (row["corruption"], float(row["severity"]), row["decoder"])
        if key in observed:
            raise ValueError("Duplicate summary condition")
        observed.add(key)
        row["severity"] = key[1]
        if int(row["n_replicates"]) != len(config["seeds"]):
            raise ValueError("Summary replicate count differs from the frozen config")
        for metric in METRICS:
            for statistic in ("mean", "std", "min", "max"):
                field = f"{metric}_{statistic}"
                row[field] = float(row[field])
                if not np.isfinite(row[field]):
                    raise ValueError("Nonfinite summary metric")
            if not (
                row[f"{metric}_min"] - 1e-12
                <= row[f"{metric}_mean"]
                <= row[f"{metric}_max"] + 1e-12
            ):
                raise ValueError("Summary mean falls outside its replicate range")
        if not 0 <= row["balanced_accuracy_min"] <= row["balanced_accuracy_max"] <= 1:
            raise ValueError("Balanced accuracy must be a fraction between zero and one")
    if observed != expected:
        raise ValueError("Summary conditions differ from the frozen severity/decoder grid")
    for decoder in DECODERS:
        if baseline[decoder]["n_trials"] != 15:
            raise ValueError("This plotting protocol requires the audited 15 held-out trials")
    return rows, config, baseline


def plot_results(report_dir: Path) -> list[Path]:
    """Save four PNG/SVG comparisons derived solely from frozen result artifacts."""
    report_dir = Path(report_dir)
    rows, config, baseline = _load_results(report_dir)
    output_dir = report_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    outputs = []
    with rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        for family, (grid, title, x_label) in FAMILIES.items():
            for metric in METRICS:
                figure = Figure(figsize=(8.8, 5.7), layout="constrained")
                FigureCanvasAgg(figure)
                axis = figure.subplots()
                is_delta = metric == "delta_balanced_accuracy"
                if is_delta:
                    axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
                else:
                    axis.axhline(50, color="#555555", linewidth=1, linestyle=":", label="Chance")
                for decoder, (label, color) in DECODERS.items():
                    series = sorted(
                        (row for row in rows if row["corruption"] == family
                         and row["decoder"] == decoder),
                        key=lambda row: row["severity"],
                    )
                    severity = np.array([row["severity"] for row in series])
                    mean = np.array([row[f"{metric}_mean"] for row in series]) * 100
                    low = np.array([row[f"{metric}_min"] for row in series]) * 100
                    high = np.array([row[f"{metric}_max"] for row in series]) * 100
                    axis.fill_between(severity, low, high, color=color, alpha=0.13, linewidth=0)
                    axis.plot(severity, mean, marker="o", color=color, linewidth=2, label=label)
                    if not is_delta:
                        original = baseline[decoder]["balanced_accuracy"] * 100
                        axis.axhline(original, color=color, linewidth=1, linestyle="--",
                                     label=f"{label} original: {original:.2f}%")
                axis.set_title(f"{title}: {'change from original' if is_delta else 'absolute performance'}",
                               fontweight="bold", pad=13)
                axis.set_xlabel(x_label, labelpad=9)
                axis.set_ylabel("Change in balanced accuracy (percentage points)" if is_delta
                                else "Balanced accuracy (%)", labelpad=9)
                axis.set_xticks(config[grid])
                if not is_delta:
                    axis.set_ylim(0, 100)
                axis.grid(axis="y", color="#D9DEE4", linewidth=0.7)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(loc="best", framealpha=0.95, fontsize=8)
                footer = (
                    f"sub-001 · run 12 · 15 trials · {len(config['seeds'])} fixed seeds\n"
                    "Lines: seed means. Bands: min–max across seeds, not confidence intervals.\n"
                    + ("Negative change means degradation; a flat near-chance control is not robustness."
                       if is_delta else "Original bandpower is near chance; a flat curve is not robustness.")
                )
                figure.supxlabel(footer, fontsize=9, color="#454B54")
                for extension in ("png", "svg"):
                    path = output_dir / f"{family}_{metric}.{extension}"
                    figure.savefig(path, dpi=180, facecolor="white")
                    outputs.append(path)
                figure.clear()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    arguments = parser.parse_args()
    for path in plot_results(arguments.report_dir):
        print(path)


if __name__ == "__main__":
    main()
