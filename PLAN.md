# BCI Stress Lab — implementation plan

Status: implementation paused at the user's request. Dependency/configuration scaffolding exists; no decoder or experiment implementation and no model training. Actual-file inspection of sub-001 runs 4, 8, and 12 is complete; see `reports/sub-001-audit.md`. A matched non-CSP decoder control is now required before interpreting stress results.

## Agreed scope

- Python, offline, CPU-only, public EEG data, reproducible experiments, tests, and a local dashboard.
- Measure both corruption severity and elapsed recording time after a corruption starts.
- Discover failure modes during development; do not impose a universal accuracy threshold now.
- Initial default: left-versus-right hand imagery, with a separately calibrated decoder for each participant.
- User-selected dataset: OpenNeuro `ds004362`, https://github.com/OpenNeuroDatasets/ds004362.git. This replaces the earlier BNCI2014-001 proposal.
- Preserve user project prompts verbatim in `prompts.md`.

Elapsed recording time is distinct from processing runtime. Offline replay can run faster than real time. This study measures controlled disturbances in recorded EEG, not time until a real headset fails.

## Minimal architecture

One Python package, one experiment configuration, file-based results, and one Streamlit viewer. No service, database, GPU, or experiment tracking server is needed.

| Component | Responsibility |
| --- | --- |
| `data.py` | Download/cache recordings; validate units, events, channels, artifact flags, and session/run/trial identity. |
| `stress.py` | Apply seeded, label-independent corruption to copied test recordings; emit a corruption timeline. |
| `pipeline.py` | Define fixed causal preprocessing and a training-only CSP/LDA estimator. |
| `experiment.py` | Make explicit splits, fit, replay, score, and save complete run artifacts. |
| `dashboard.py` | Read saved results and display severity and time effects. |

Use a small explicit loader for this OpenNeuro dataset with MNE's EEGLAB reader, MNE/SciPy for signal processing, NumPy and scikit-learn for decoding, Pydantic for configuration, pytest for tests, and Streamlit for the dashboard. MOABB is unnecessary for the first version. Keep configuration under `configs/`, results under `runs/`, and cached recordings outside version control. Pin Python and dependencies in a lockfile.

## Phase 1 — Establish the data contract

Use OpenNeuro `ds004362`, a BIDS/EEGLAB conversion of the PhysioNet EEG Motor Movement/Imagery Dataset. The repository contains 109 participant directories; the protocol specifies 64 EEG channels and 14 runs. Actual sub-001 SET files for runs 4, 8, and 12 have been downloaded and validated against pinned annex hashes and source EDFs: all are 64 × 20,000 continuous EEG samples at 160 Hz, 125 seconds, with 15 imagery trials per run and no dedicated EOG. Audit these properties across other selected recordings rather than assuming uniformity.

Select original protocol runs 4, 8, and 12 only: left/right hand imagery. Parse run numbers numerically. All BIDS filenames use `task-motion`, so that filename alone does not distinguish imagery from execution. In the converted files, map `TASK2T1` to left imagery, `TASK2T2` to right imagery, and `TASK2T0` to rest. Keep rest as replay context but not a classification target. Exclude other task types explicitly.

Pin an OpenNeuro snapshot and matching repository commit before downloading; the dataset metadata currently identifies DOI `10.18112/openneuro.ds004362.v1.0.0`. Retrieve only the required participant/run EEG files and metadata initially. EEG `.set` files are git-annex links: an ordinary Git clone alone does not fetch their signal content. Use DataLad/git-annex or the OpenNeuro download route, verify actual files and checksums, and include any companion signal file required by the format. Keep converted `.set` files as the canonical input; use original EDF files in `sourcedata` for conversion/units cross-checks, never as additional independent observations.

Verify cue timing, sampling rate, channel order, volts versus microvolts, original reference, and any available artifact annotations against the recording headers and documentation. The inspected channel TSV leaves both channel type and units as `n/a`, and its event TSV leaves duration as `n/a`. Source EDF comparisons establish that embedded SET values are microvolts and MNE imports are volts, identical sample-for-sample to EDF imports. SET event latency is one-based; TSV samples are zero-based. Event durations are missing in SET but are 4.1 s for imagery and 4.2 s for rest in source EDF. In all three audited runs, the last imagery ends at 124.5 s and the final 0.5 s is unannotated: do not use file end as a substitute imagery offset. Store source-verified durations in the audit manifest. A classification window must stay within its imagery interval. Missing metadata must never silently become fabricated units, durations, or artifact labels. Maintain stable trial identifiers. Record every exclusion and its reason. Call unchanged data “original,” not “clean.”

Tradeoff: broader participant coverage, but only three short runs for this particular binary imagery task, giving relatively few trials per participant. The proposed evaluation is later-run generalization, not next-day generalization. Short runs limit time-to-failure resolution; there is no basis for hours-long fatigue or electrode-drift claims. Cite the OpenNeuro dataset and original study, and preserve license/provenance metadata (the inspected OpenNeuro description declares CC0). The original PhysioNet release is the same underlying data and is not an independent replication dataset.

Acceptance: a data audit lists participants, runs, trial counts by class, channel metadata, and exclusions, with verified event-window alignment.

## Phase 2 — Establish a baseline and explore failures

Preprocessing default: preserve acquisition reference and sampling rate; use EEG channels only; apply a fixed fourth-order causal Butterworth 8–30 Hz bandpass in second-order sections to each continuous run; initially propose the half-open interval from 1 to 4 seconds after each imagery cue. Validate that window against actual event transitions during the data audit; if it is unsuitable, choose a common shorter window using development recordings and document it before testing. Fix initialization and run-start warm-up rules before evaluation. Do not filter across run boundaries or use future samples.

Decoder A: four regularized CSP components, log-power features, and shrinkage LDA in one fitted pipeline. Mandatory decoder B: fixed per-channel log-bandpower on the exact same channels and temporal band, followed by the same shrinkage LDA settings, with no learned spatial mixing. Fix settings initially. Use a training-defined or preregistered power floor to handle flatlines without undefined logarithms. CSP/LDA is inexpensive and established, but can be sensitive to covariance and channel changes. A deep network would add tuning without helping establish correctness.

Frozen CSP does not re-estimate covariance on corrupted test signals. Comparing A and B probes sensitivity associated with the learned spatial feature pipeline; B still uses feature covariance through LDA and has a different feature dimension, so this is not a pure isolation of all covariance dependence. Show both absolute balanced accuracy and paired baseline-subtracted degradation. A control near chance cannot establish useful robustness from a flat degradation curve. See `reports/decoder-controls.md` for the full control design.

Avoid ICA, automatic channel interpolation, and amplitude-based rejection initially. They would change which disturbances the decoder sees. This dataset does not provide dedicated EOG in the inspected recordings; do not present synthetic noise as validated ocular contamination. SET says `ref='common'` while sidecars say “Left or right ear lobe”; preserve the original signals and do not infer common-average rereferencing. Record any known acquisition/conversion filtering and mark undocumented details unknown.

Use run 4 for development training and run 8 for development validation, separately for each participant. Keep run 12 sealed for final evaluation. Fit CSP, LDA, and any learned scales only on the development training split. No overlapping windows or trials may cross a split. Few trials and 64 channels make covariance estimation fragile, so use fixed covariance regularization and show uncertainty rather than conducting a broad hyperparameter search.

Acceptance: a reproducible baseline report for all participants, plus exploratory notes on failure modes. Baseline accuracy is a finding, not a hardcoded test expectation.

## Phase 3 — Measure severity sensitivity

Freeze each fitted decoder during stress evaluation. Start with two stressors, tested separately:

1. Additive broadband Gaussian noise. Specify noise RMS as a multiple of a fixed per-channel scale estimated from training recordings; also save actual amplitude in microvolts.
2. Channel flatlining. Replace selected EEG channels with zero in the recorded reference, preserving shape and channel order. This is a defined fault proxy, not a complete model of loose electrodes.

Corrupt copies of continuous recordings before filtering. Use original and stressed versions of the same trials, and pass the exact same corrupted arrays to both decoder arms. Choose a small severity grid during development, include a zero-corruption control, and use ten fixed corruption seeds. Reuse the underlying noise realization across amplitudes and nested channel selections across dropout counts. Never condition faults on class labels.

After the two baselines are usable, add two small diagnostics: (1) a single-channel fault sweep, with anatomical groupings fixed before scoring, to separate fault count from location; (2) independently corrupted training-data refits at selected known fault conditions, without fitting any test data. Recovery supports calibration mismatch; lack of recovery can reflect information loss or model limitations and is not proof of irrecoverable information loss. Keep adaptation diagnostics separate from frozen-decoder robustness.

Measure balanced accuracy, paired percentage-point change from the original condition, class recalls, and confusion matrices. Report each participant and an equally weighted participant mean. Keep variation across participants separate from variation across corruption seeds; do not treat seeds as additional participants. Capture numerical failures and missing predictions explicitly rather than dropping them from metrics.

Acceptance: paired performance-versus-severity curves, with corruption metadata sufficient to regenerate every condition.

## Phase 4 — Measure elapsed-time effects through replay

Replay recordings chronologically in chunks without wall-clock sleeps. Preserve causal filter state within each run and reset only at documented boundaries. Keep the decoder frozen. Emit one prediction per eligible imagery trial once its complete classification window has arrived.

Use two schedules:

- Step exposure: original signal, then a fixed corruption severity for a specified duration, then removal. Compare several severities, exposure durations, and label-independent onsets to inspect degradation and filter recovery without relying on one particular sequence of classes.
- Ramp exposure: severity increases with elapsed recording time. Show current severity alongside time; a ramp alone cannot distinguish time effects from severity effects.

Set onset schedules independently of labels. Save run-relative timestamps, time since onset, affected channels, severity, and trial-window exposure. Do not invent continuous session time across unknown gaps between runs: the audited EDF headers have identical date/time values and cannot establish those gaps. Label trials overlapping onset or removal separately. A frozen trial-wise decoder does not accumulate fatigue; time effects here can arise from filter transients and changing EEG/trial composition. With only 15 imagery trials per run, report insufficient evidence rather than forcing a precise time-to-failure estimate.

Plot per-trial errors and rolling performance with the window size, class counts, and denominator visible. Do not score rest intervals or pretend the cued dataset supplies continuous motor-intent ground truth. Report processing runtime separately. If failure thresholds are later selected, name the rolling rule and persistence requirement, and report “already below threshold,” “not reached,” or “insufficient observations” as applicable. Detection resolution is limited by trial timing and aggregation windows.

Acceptance: a timeline showing disturbance onset, exposure, predictions, and recovery, with chunked replay matching whole-run causal processing within numerical tolerance.

## Phase 5 — Freeze and confirm

Failure definitions can evolve during run-4/run-8 exploration. Record each chosen operational definition, severity grid, and time schedule before opening run-12 results. Refit on runs 4 and 8 together and evaluate run 12 under that frozen protocol. This is a within-participant, held-out-run test. Do not choose exclusions, model settings, or thresholds to improve run-12 outcomes. If a required run is missing or fails objective data checks, report the participant as ineligible for that protocol rather than substituting a different task. Any subsequent redesign is explicitly exploratory and needs fresh data for independent confirmation.

## Reproducibility and tests

Every run saves validated configuration, seeds, source checksums, exact splits, Git revision and dirty status, dependency versions, model parameters, corruption timelines, trial-level predictions, errors, and aggregate metrics. Keep outputs separate per run and make interrupted runs visibly incomplete.

Tests cover event timing and units; explicit TASK2 label mapping and rejection of execution/other imagery tasks; numeric run ordering; missing-duration handling and windows staying inside imagery intervals; split isolation and training-only fitting; unresolved annex links; zero-corruption identity; deterministic corruption and source immutability; noise amplitude and channel targeting; causal filtering with no future-sample dependence; replay chunk equivalence; and metric behavior with missing classes or invalid predictions. Use synthetic fixtures for fast offline tests and a small cached real-data integration test. A shuffled-label development check is a diagnostic for leakage, not a brittle exact-accuracy assertion.

## Phase 6 — Local dashboard and first release

The Streamlit app reads completed artifacts without fitting models. Provide run and participant selectors, original baseline, severity curves, replay timelines, confusion matrices, and original/stressed EEG snippets. Show protocol, seeds, exclusions, missing predictions, and exploratory-versus-confirmatory status. It must work offline after data download.

Deliver first a one-participant baseline-to-dashboard slice using sub-001 if it passes data checks, then a fixed small pilot selected by participant ID, then all eligible participants from the 109 listed in the repository under the same frozen protocol. Do not select participants by decoding accuracy. Process one participant at a time to bound CPU memory use. The first release is complete when a documented command regenerates the experiment artifacts, required tests pass, and the dashboard exposes both severity and elapsed-time results. Add recorded ocular/movement artifacts or an independent second dataset only after this path is verified.

## References

- Selected repository, protocol, and event labels: https://github.com/OpenNeuroDatasets/ds004362
- Dataset metadata and DOI: https://github.com/OpenNeuroDatasets/ds004362/blob/main/dataset_description.json
- Inspected recording metadata: https://github.com/OpenNeuroDatasets/ds004362/blob/main/sub-001/eeg/sub-001_task-motion_run-4_eeg.json
- Inspected event table: https://github.com/OpenNeuroDatasets/ds004362/blob/main/sub-001/eeg/sub-001_task-motion_run-4_events.tsv
- Inspected channel table: https://github.com/OpenNeuroDatasets/ds004362/blob/main/sub-001/eeg/sub-001_task-motion_run-4_channels.tsv
- CSP/LDA reference implementation: https://mne.tools/1.12/auto_examples/decoding/decoding_csp_eeg.html
- Original underlying dataset: https://physionet.org/content/eegmmidb/1.0.0/
