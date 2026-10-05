# Source EDF and annex provenance audit

Audit performed 2026-10-05. Parsed actual EDF bytes directly with Python standard-library code; no decoder fitting or experiment implementation. Full machine-readable headers and TAL events: `edf-provenance-audit.json`.

## Provenance

Pinned repository commit: [`af0c461f23e8a7aa782475e3c22d160bc39eb170`](https://github.com/OpenNeuroDatasets/ds004362/tree/af0c461f23e8a7aa782475e3c22d160bc39eb170). Downloaded SHA256E annex pointer contents from this commit and compared payload SHA-256 and byte lengths. All six files match.

| File | Bytes | SHA-256 | Match |
|---|---:|---|---|
| `sourcedata/rawdata/S001/S001R04.edf` | 2596896 | `3d161f88e1c00632585287d2ce584c2bc0f08862438eb255ea8723e00fac693d` | True |
| `sourcedata/rawdata/S001/S001R08.edf` | 2596896 | `358fb5189220725141968ae285fbe9e3f36210b834ffba71d940af308e3aca68` | True |
| `sourcedata/rawdata/S001/S001R12.edf` | 2596896 | `2b281c9b687b4c4176e83251d74743721f2d6ebd76656a972a3b9c44d9d88cd5` | True |
| `sub-001/eeg/sub-001_task-motion_run-4_eeg.set` | 5935688 | `b3f2d5148bc92a4985590536996e7bf87eb44e0a09b2b66059682e13346ab31c` | True |
| `sub-001/eeg/sub-001_task-motion_run-8_eeg.set` | 5935688 | `d56223014c405fb3a8994028f0100a4ed6c939988802a9ae920be5fe177e9d4a` | True |
| `sub-001/eeg/sub-001_task-motion_run-12_eeg.set` | 5935688 | `3a802988d8adaee922e5d8797698bb8e118739823bae61dd6403ec7dd4494ac7` | True |

## EDF headers

All three EDF files share the following properties:

- EDF+C, 125 continuous data records, 1 second per record; exactly 125 seconds and 20,000 EEG samples per channel. Last EEG sample is at 124.99375 seconds.
- 65 signals: 64 EEG channels sampled at 160 Hz plus `EDF Annotations` containing 80 16-bit slots per record. The annotation storage rate is not an EEG sample rate.
- Header length 16,896 bytes; total length 2,596,896 bytes exactly agrees with header + data-record sizes.
- EEG physical dimension is `uV`; EEG physical minimum/maximum = digital minimum/maximum = -8092 / +8092. Thus each stored integer equals that many microvolts, with no offset. Convert to volts by multiplying by 1e-6.
- Annotation channel physical dimension `-`, minima/maxima -32768 / +32767.
- Prefilter field is literally `HP:0Hz LP:0Hz N:0Hz`. This field alone does not establish the complete acquisition/filtering history.
- Each EDF starts at header date `12.08.09`, time `16.15.00`; identical recorded start times cannot establish actual elapsed time between runs.

Exact EDF EEG channel labels, in data order:

```text
Fc5., Fc3., Fc1., Fcz., Fc2., Fc4., Fc6., C5.., C3.., C1.., Cz.., C2.., C4.., C6.., Cp5., Cp3., Cp1., Cpz., Cp2., Cp4., Cp6., Fp1., Fpz., Fp2., Af7., Af3., Afz., Af4., Af8., F7.., F5.., F3.., F1.., Fz.., F2.., F4.., F6.., F8.., Ft7., Ft8., T7.., T8.., T9.., T10., Tp7., Tp8., P7.., P5.., P3.., P1.., Pz.., P2.., P4.., P6.., P8.., Po7., Po3., Poz., Po4., Po8., O1.., Oz.., O2.., Iz..
```

Channel name normalization (e.g. `C3..` to `C3`) is a separate metadata operation; preserve and verify the mapping. No EOG channel is present in these EDF headers.

## Annotations

The following table contains every annotated task onset in seconds from recording start. EDF TAL annotations use `T0`, `T1`, `T2`. Interpretation as rest/left imagery/right imagery comes from run-specific task documentation, not from the bare labels. Rest intervals and imagery intervals alternate.

| Index | Rest onset (T0), all runs | Imagery onset, all runs | Run 4 label | Run 8 label | Run 12 label |
|---:|---:|---:|---|---|---|
| 1 | 0.0 | 4.2 | T2 | T1 | T2 |
| 2 | 8.3 | 12.5 | T1 | T2 | T1 |
| 3 | 16.6 | 20.8 | T1 | T1 | T2 |
| 4 | 24.9 | 29.1 | T2 | T2 | T1 |
| 5 | 33.2 | 37.4 | T2 | T1 | T1 |
| 6 | 41.5 | 45.7 | T1 | T2 | T2 |
| 7 | 49.8 | 54.0 | T2 | T1 | T2 |
| 8 | 58.1 | 62.3 | T1 | T2 | T1 |
| 9 | 66.4 | 70.6 | T2 | T2 | T1 |
| 10 | 74.7 | 78.9 | T1 | T1 | T2 |
| 11 | 83.0 | 87.2 | T1 | T1 | T2 |
| 12 | 91.3 | 95.5 | T2 | T2 | T1 |
| 13 | 99.6 | 103.8 | T1 | T2 | T2 |
| 14 | 107.9 | 112.1 | T2 | T1 | T1 |
| 15 | 116.2 | 120.4 | T1 | T1 | T2 |

Every T0 duration is 4.2 seconds; every T1/T2 duration is 4.1 seconds. Each file also has 125 unlabeled record-timekeeping TAL entries, which must not be counted as task events.

| Run | T0 rest states | T1 | T2 | Imagery trials |
|---|---:|---:|---:|---:|
| 4 | 15 | 8 | 7 | 15 |
| 8 | 15 | 8 | 7 | 15 |
| 12 | 15 | 7 | 8 | 15 |

## Implications and cautions

- Only 15 imagery trials per run (7/8 per class), not 30 imagery trials: 30 includes rest states. Pilot performance and time-to-failure estimates will be coarse and unstable.
- Source annotation intervals end at 124.5 seconds. The remaining 0.5 seconds are recorded but have no explicit task label.
- A 1–4 second post-cue epoch fits inside each 4.1-second imagery annotation, including the last trial. Causal filtering should nevertheless run over the complete continuous run, separately per run, with initialization behavior defined.
- Dataset metadata inspection verifies that expected cue labels are present; it cannot prove participant adherence or successful imagery.
- These runs support short offline playback experiments. The repeated date/time headers do not support connecting them into a trustworthy continuous elapsed-time series.
- Inspecting held-out run 12 labels/timing is data validation, not model fitting. Keep predictions and decoder-dependent decisions away from run 12 until the protocol is frozen.
