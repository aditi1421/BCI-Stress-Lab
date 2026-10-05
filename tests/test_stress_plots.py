"""Plots require complete frozen outputs and never rerun decoding."""

import csv
import json

import pytest

from bci_stress.plot_stress import plot_results


@pytest.fixture
def completed_results(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete"}))
    config = {"noise_levels": [0, 0.25], "dropout_counts": [0, 1], "seeds": [0, 1]}
    (tmp_path / "stress_config.json").write_text(json.dumps(config))
    baseline = {
        "csp_lda": {"n_trials": 15, "balanced_accuracy": 0.6},
        "bandpower_lda": {"n_trials": 15, "balanced_accuracy": 0.46},
    }
    (tmp_path / "clean_baseline_metrics.json").write_text(json.dumps(baseline))
    rows = []
    for family, grid in (("gaussian", "noise_levels"), ("flatline", "dropout_counts")):
        for severity in config[grid]:
            for decoder, scores in baseline.items():
                row = {"corruption": family, "severity": severity, "decoder": decoder,
                       "n_replicates": 2}
                for metric in ("balanced_accuracy", "delta_balanced_accuracy"):
                    mean = scores["balanced_accuracy"] if metric == "balanced_accuracy" else 0
                    row.update({f"{metric}_mean": mean, f"{metric}_std": 0,
                                f"{metric}_min": mean, f"{metric}_max": mean})
                rows.append(row)
    with (tmp_path / "summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_plots_all_four_comparisons_from_completed_outputs(completed_results):
    source_bytes = {path.name: path.read_bytes() for path in completed_results.iterdir()}
    plots = plot_results(completed_results)
    assert len(plots) == 8
    assert all(path.stat().st_size > 1000 for path in plots)
    assert {path.stem for path in plots} == {
        "gaussian_balanced_accuracy", "gaussian_delta_balanced_accuracy",
        "flatline_balanced_accuracy", "flatline_delta_balanced_accuracy",
    }
    svg = (completed_results / "plots/gaussian_balanced_accuracy.svg").read_text()
    assert "not confidence intervals" in svg
    assert "Original bandpower is near chance" in svg
    for name, original in source_bytes.items():
        assert (completed_results / name).read_bytes() == original


@pytest.mark.parametrize("status", ["running", "failed"])
def test_rejects_incomplete_results(completed_results, status):
    (completed_results / "status.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match="completed"):
        plot_results(completed_results)
    assert not (completed_results / "plots").exists()


def test_rejects_missing_summary_condition(completed_results):
    summary = completed_results / "summary.csv"
    summary.write_text("\n".join(summary.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="grid"):
        plot_results(completed_results)
