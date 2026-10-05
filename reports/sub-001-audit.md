# Actual-file audit: ds004362, sub-001, runs 4, 8, 12

Inspected 2026-10-05. This is a data audit, not a training result. The three complete `.set` files were read directly with SciPy and MNE and compared sample-for-sample with the original EDFs from this dataset. All six binaries match the SHA256 and size in git-annex pointers at repository commit `af0c461f23e8a7aa782475e3c22d160bc39eb170`; see `edf-provenance-audit.md` and its JSON manifest.

## Contents common to all three SET files

| Field | Actual value |
| --- | --- |
| Storage | MATLAB 5-format EEGLAB file, numeric data embedded; no external `.fdt` required |
| Signal array | `64 × 20000`, float32, all finite |
| Sampling rate | 160 Hz; sample interval 0.00625 s |
| Recording length | 125 seconds (`20000 / 160`) |
| First/last sample times | 0 and 124.99375 seconds |
| `times` field | Milliseconds: 0 through 124993.75 |
| EEGLAB `trials` field | 1: one continuous recording, not one imagery trial |
| Session | `session = 1` in all three files |
| Run | `run = 4`, `8`, or `12`, correctly matching the filename |
| Channel types | 64 EEG by source/protocol; SET channel `type` fields empty and TSV type/units `n/a` |
| Dedicated EOG/EMG/ECG | None in these files |
| Signal units | Embedded numeric values are microvolts, verified against EDF physical scaling; MNE returns volts |
| Reference | SET `ref = 'common'`; sidecar says “Left or right ear lobe”; neither identifies which ear |
| Event labels | `TASK2T0`, `TASK2T1`, `TASK2T2` only in `event`; original `T0/T1/T2` retained in `urevent` |
| Event count | 30: 15 rest onsets and 15 imagery onsets |
| Event durations | Empty in SET; `n/a` in TSV; available in original EDF annotations |
| Processing fields | ICA matrices empty, epoch array empty, rejection flag arrays empty; no artifact labels to rely on |

The 64 channel names below are the exact stored spelling and order, shared by all three files:

```text
Fc5, Fc3, Fc1, Fcz, Fc2, Fc4, Fc6, C5,
C3, C1, Cz, C2, C4, C6, Cp5, Cp3,
Cp1, Cpz, Cp2, Cp4, Cp6, Fp1, Fpz, Fp2,
Af7, Af3, Afz, Af4, Af8, F7, F5, F3,
F1, Fz, F2, F4, F6, F8, Ft7, Ft8,
T7, T8, T9, T10, Tp7, Tp8, P7, P5,
P3, P1, Pz, P2, P4, P6, P8, Po7,
Po3, Poz, Po4, Po8, O1, Oz, O2, Iz
```

EDF labels have trailing dots and different casing. After normalizing only casing and trailing dots, all 64 labels match in order. No channel reordering was needed for signal comparison.

## Units and conversion verification

The EDF EEG headers explicitly specify `uV`, physical minimum/maximum `-8092 / 8092`, and digital minimum/maximum `-8092 / 8092`: one digital unit is one microvolt, with zero offset. For every sample and channel in all three recordings, MNE's SET and EDF imports are identical in volts (maximum absolute difference 0 V). SET numeric values multiplied by `1e-6` also match exactly. This verifies the units without relying on the incomplete channel TSV.

| Run | Embedded value range (µV) | Left imagery | Right imagery | Rest onsets |
| --- | --- | --- | --- | --- |
| 4 | -376 to 595 | 8 | 7 | 15 |
| 8 | -507 to 582 | 8 | 7 | 15 |
| 12 | -494 to 586 | 7 | 8 | 15 |

These are whole-recording extrema, not artifact classifications. Exact signal equality also shows that this conversion did not change the numeric EEG samples; it does not establish that the original acquisition was unfiltered.

## Exact event timing

`TASK2T0` means rest, `TASK2T1` left-fist imagery, and `TASK2T2` right-fist imagery. No TASK1 execution or TASK4 hands/feet events occur in these files. Their task labels and HED descriptions identify the expected imagery condition; they do not prove participant compliance.

SET `event.latency` uses one-based sample indices. Convert with `(latency - 1) / 160`. TSV `sample` is zero-based. These conversions match all TSV and source EDF onsets exactly for all 90 events. For example, SET latency 673 = TSV sample 672 = 4.2 seconds.

The following table contains every imagery onset. L = `TASK2T1`, R = `TASK2T2`. EDF duration is 4.1 seconds for every row in every run.

| Trial | Onset (s) | End from EDF (s) | Run 4 | Run 8 | Run 12 |
| --- | --- | --- | --- | --- | --- |
| 1 | 4.2 | 8.3 | R | L | R |
| 2 | 12.5 | 16.6 | L | R | L |
| 3 | 20.8 | 24.9 | L | L | R |
| 4 | 29.1 | 33.2 | R | R | L |
| 5 | 37.4 | 41.5 | R | L | L |
| 6 | 45.7 | 49.8 | L | R | R |
| 7 | 54.0 | 58.1 | R | L | R |
| 8 | 62.3 | 66.4 | L | R | L |
| 9 | 70.6 | 74.7 | R | R | L |
| 10 | 78.9 | 83.0 | L | L | R |
| 11 | 87.2 | 91.3 | L | L | R |
| 12 | 95.5 | 99.6 | R | R | L |
| 13 | 103.8 | 107.9 | L | R | R |
| 14 | 112.1 | 116.2 | R | L | L |
| 15 | 120.4 | 124.5 | L | L | R |

Rest onsets are identical across runs: **0, 8.3, 16.6, 24.9, 33.2, 41.5, 49.8, 58.1, 66.4, 74.7, 83.0, 91.3, 99.6, 107.9, 116.2 seconds**. Every rest annotation lasts 4.2 seconds and ends at the following imagery onset. The final interval **124.5–125.0 seconds has no annotated task state**. Do not extend the last imagery event to file end.

All 45 half-open windows `[cue + 1 s, cue + 4 s)` fit inside their source-annotated imagery intervals. Each contains 480 samples. The final trial's window ends at 124.4 s, before the 124.5 s imagery offset.

## Corrections and limits for the plan

1. **Missing event durations are consequential.** Copying next-transition durations mostly works, but treating file end as the final imagery offset incorrectly adds 0.5 s. Use verified source EDF durations for the audit manifest and mark the tail unknown.
2. **Units must be explicit.** Stored SET values are µV; MNE delivers V. Do not inject µV amplitudes directly into a volts array. Preserve the verified conversion in the data contract.
3. **“Common” does not establish average reference.** The source EEG is numerically unchanged; do not interpret this SET field as evidence of common-average rereferencing. Preserve the recorded signals and retain the ear-reference ambiguity in provenance.
4. **Only 15 training trials in run 4.** Run 8 supplies 15 development trials; final training on 4+8 supplies 30. There are 64 channels. Regularization matters, and fine-grained time-to-failure claims will have very little evidence per window. One additional class error changes balanced accuracy by about 6.25 or 7.14 percentage points in these runs.
5. **No across-day experiment here.** All three files say session 1. Run-12 evaluation is within-participant, later-run testing, not cross-session/day transfer.
6. **Elapsed time is short and sparsely labeled.** Each run lasts 125 s, with one imagery onset every 8.3 s. Replay can show disturbance response and recovery, but not long-term fatigue/drift or dense continuous intent decoding. All three EDF headers have the same date/time (`12.08.09`, `16.15.00`), which cannot establish inter-run gaps. Do not concatenate runs as uninterrupted physiological time.
7. **No dedicated EOG or artifact ground truth.** Synthetic channel faults measure controlled sensitivity; realism requires a later empirical artifact validation.
8. **CSP cannot be the only decoder.** Add matched fixed-channel log-bandpower + shrinkage LDA as a mandatory control; see `decoder-controls.md`. Similar or different deterioration across these two pipelines does not establish universal BCI fragility.

The selected task, run split, and 1–4 s windows are supported by these files. During this audit, run 12 was inspected for data integrity only; no model was fit or scored and no performance-driven choice was made. The subsequent fixed clean-decoding evaluation is documented in the [completed result report](clean-sub001/README.md).

## Saved evidence

- `sub-001-file-audit.json`: machine-readable SET/header/unit comparisons.
- `sub-001-events.csv`: all 90 events, both indexing conventions, onsets, source durations and ends, and epoch eligibility.
- `edf-provenance-audit.md` / `.json`: independent EDF parsing and pinned SHA256 checks.
- `decoder-controls.md`: paired decoder experiment and interpretation limits.

Source protocol: https://github.com/OpenNeuroDatasets/ds004362/tree/af0c461f23e8a7aa782475e3c22d160bc39eb170
