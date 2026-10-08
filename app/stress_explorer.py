"""Interactive explorer for the frozen sub-001 corruption experiment.

Reads only committed result files under reports/stress-sub001. When the audited
EEG is cached under data/, it also redraws the held-out recording with the same
seeded corruption the experiment applied. Nothing here refits or reruns a decoder.

    uv run --group ui streamlit run app/stress_explorer.py
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

from bci_stress.corruption import add_gaussian_noise, flatline_channels

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "stress-sub001"
TEST_RECORDING = ROOT / "data/ds004362/sub-001/eeg/sub-001_task-motion_run-12_eeg.set"
SAMPLING_HZ = 160
MOTOR_CHANNELS = ("C3", "Cz", "C4")
DECODERS = {"csp_lda": "CSP + LDA", "bandpower_lda": "Bandpower + LDA"}
FAMILIES = {
    "gaussian": ("Gaussian noise", "× training RMS"),
    "flatline": ("Channel flatline", "channels"),
}


@st.cache_data
def load_results():
    metrics = pd.read_csv(REPORT / "metrics.csv")
    metrics["failed_channels"] = metrics["failed_channels"].map(json.loads)
    predictions = pd.read_csv(REPORT / "predictions.csv")
    summary = pd.read_csv(REPORT / "summary.csv")
    scale = json.loads((REPORT / "training_scale.json").read_text())
    return metrics, predictions, summary, scale


@st.cache_data
def load_test_recording():
    if not TEST_RECORDING.exists():
        return None
    recording = loadmat(TEST_RECORDING, simplify_cells=True)
    return np.array(recording.get("EEG", recording)["data"], dtype=np.float64)


def severity_label(family: str, severity: float) -> str:
    unit = FAMILIES[family][1]
    return f"{severity:g}{unit}" if family == "gaussian" else f"{int(severity)} {unit}"


def corrupt(native, family, severity, seed, failed, scale):
    if family == "gaussian":
        return add_gaussian_noise(native, np.asarray(scale["rms_uv"]), float(severity), seed)
    if not failed:
        return native.copy()
    corrupted, _ = flatline_channels(native, scale["channels"], names=failed)
    return corrupted


st.set_page_config(page_title="BCI Stress Lab", layout="wide")
metrics, predictions, summary, scale = load_results()

st.title("BCI Stress Lab")
st.caption(
    "Two motor imagery decoders, trained once on sub-001 runs 4 and 8 and then frozen, "
    "classify 15 held-out left/right trials from run 12 while that recording is corrupted."
)

with st.sidebar:
    st.header("Corruption")
    family = st.radio(
        "Family", list(FAMILIES), format_func=lambda key: FAMILIES[key][0],
    )
    severities = sorted(metrics.loc[metrics["corruption"] == family, "severity"].unique())
    severity = st.select_slider(
        "Severity", severities, value=severities[0],
        format_func=lambda value: severity_label(family, value),
    )
    seed = st.slider("Seed", 0, 9, 0)
    st.divider()
    st.caption(
        "Severities and seeds were frozen before any nonzero score was produced. "
        "Seed spread describes perturbations of the same 15 trials, not population uncertainty."
    )

condition = metrics[
    (metrics["corruption"] == family) & (metrics["severity"] == severity) & (metrics["seed"] == seed)
]
failed = condition["failed_channels"].iloc[0]

# Decoder scorecards
columns = st.columns(len(DECODERS))
for column, (decoder, name) in zip(columns, DECODERS.items()):
    row = condition[condition["decoder"] == decoder].iloc[0]
    with column:
        st.metric(
            f"{name} balanced accuracy",
            f"{row['balanced_accuracy']:.1%}",
            f"{row['delta_balanced_accuracy'] * 100:+.2f} pp vs original" if severity else None,
            delta_color="off",
        )
        if row["constant_prediction"]:
            side = "left" if row["constant_class"] == 0 else "right"
            st.error(f"Collapsed: predicts **{side}** for all 15 trials")
        else:
            st.caption(
                f"Left recall {row['left_recall']:.0%} · right recall {row['right_recall']:.0%}"
            )

# Held-out EEG under the selected corruption
st.subheader("Held-out EEG, run 12")
native = load_test_recording()
trials = predictions[
    (predictions["condition_id"] == condition["condition_id"].iloc[0])
    & (predictions["decoder"] == "csp_lda")
].reset_index(drop=True)
if native is None:
    st.info("Cache the audited ds004362 recordings under data/ to draw the corrupted signal.")
else:
    trial = st.selectbox(
        "Trial", trials.index,
        format_func=lambda i: f"Trial {i + 1}: imagined {'left' if trials.at[i, 'true_label'] == 0 else 'right'} hand",
    )
    start, stop = int(trials.at[trial, "start_sample"]), int(trials.at[trial, "stop_sample"])
    corrupted = corrupt(native, family, severity, seed, failed, scale)
    shown = list(dict.fromkeys([*MOTOR_CHANNELS, *failed[:3]]))
    index = [scale["channels"].index(name) for name in shown]
    time_s = np.arange(stop - start) / SAMPLING_HZ + 1.0
    frames = []
    for label, values in (("corrupted", corrupted), ("original", native)):
        for name, i in zip(shown, index):
            frames.append(pd.DataFrame({
                "seconds after cue": time_s, "µV": values[i, start:stop],
                "channel": name, "signal": label,
            }))
    traces = pd.concat(frames)
    chart = alt.Chart(traces).mark_line(strokeWidth=1).encode(
        x="seconds after cue:Q",
        y=alt.Y("µV:Q", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "signal:N", scale=alt.Scale(domain=["original", "corrupted"], range=["#1f2933", "#f08a96"]),
        ),
    ).properties(height=110, width=820).facet(row=alt.Row("channel:N", sort=shown, title=None))
    st.altair_chart(chart, use_container_width=True)
    if failed:
        st.caption(f"Flatlined: {', '.join(failed)}")

# Per-trial predictions
st.subheader("Trial by trial")
for decoder, name in DECODERS.items():
    rows = predictions[
        (predictions["condition_id"] == condition["condition_id"].iloc[0])
        & (predictions["decoder"] == decoder)
    ].reset_index(drop=True)
    rows["trial"] = rows.index + 1
    rows["outcome"] = np.where(rows["true_label"] == rows["predicted_label"], "correct", "wrong")
    rows["truth"] = np.where(rows["true_label"] == 0, "L", "R")
    tiles = alt.Chart(rows).mark_square(size=900, opacity=1).encode(
        x=alt.X("trial:O", title=None),
        color=alt.Color(
            "outcome:N", scale=alt.Scale(domain=["correct", "wrong"], range=["#2e9e6b", "#d1495b"]),
            legend=None,
        ),
        tooltip=["trial", "truth", alt.Tooltip("probability_right:Q", format=".2f")],
    )
    text = alt.Chart(rows).mark_text(color="white", fontWeight="bold").encode(
        x=alt.X("trial:O", title=None), text="truth:N",
    )
    st.caption(f"{name}: tile shows the true hand, green when predicted correctly")
    st.altair_chart((tiles + text).properties(height=48), use_container_width=True)

# Degradation curve across all seeds
st.subheader("Across all ten seeds")
curve = summary[summary["corruption"] == family].copy()
curve["decoder"] = curve["decoder"].map(DECODERS)
decoder_color = alt.Color(
    "decoder:N", title=None,
    scale=alt.Scale(domain=list(DECODERS.values()), range=["#3b6fb6", "#e08a1e"]),
)
accuracy = alt.Scale(domain=[0.3, 0.8], clamp=True)
x = alt.X("severity:Q", title=FAMILIES[family][0] + f" ({FAMILIES[family][1]})")
band = alt.Chart(curve).mark_area(opacity=0.15).encode(
    x=x, y=alt.Y("balanced_accuracy_min:Q", scale=accuracy), y2="balanced_accuracy_max:Q",
    color=decoder_color,
)
line = alt.Chart(curve).mark_line(point=True).encode(
    x=x,
    y=alt.Y("balanced_accuracy_mean:Q", title="balanced accuracy", scale=accuracy),
    color=decoder_color,
)
chance = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="y:Q")
marker = alt.Chart(pd.DataFrame({"severity": [severity]})).mark_rule(color="#d1495b").encode(x="severity:Q")
st.altair_chart((band + line + chance + marker).properties(height=300), use_container_width=True)
st.caption(
    "Shaded bands span the minimum to maximum across seeds. The dashed line is chance. "
    "A flat near-chance curve is not useful robustness."
)
