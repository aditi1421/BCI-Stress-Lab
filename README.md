# BCI Stress Lab

An offline, CPU-only motor-imagery decoding baseline using audited OpenNeuro **ds004362**, participant **sub-001**. This release implements the original-signal decoding stage. The project name describes the planned later stress experiments.

The [first fixed evaluation](reports/clean-sub001/README.md) gives **60.71% balanced accuracy for CSP + LDA** and **46.43% for channel bandpower + LDA** on 15 held-out trials. Both scores lie within their respective label-permutation reference ranges. The near-chance control and small sample limit what can be inferred. The frozen run reproduced exactly.

## Fixed experiment

| Setting | Value |
| --- | --- |
| Training | Runs 4 and 8: 30 imagery epochs (16 left, 14 right) |
| Later-run test | Run 12: 15 imagery epochs (7 left, 8 right) |
| Channels | All 64 EEG channels, original order and recorded reference |
| Signal units | Volts internally, validated sample-for-sample against original EDF |
| Preprocessing | 8–30 Hz causal Butterworth SOS, prototype order 4; zero state separately per continuous run |
| Epochs | Half-open `[cue + 1, cue + 4)` seconds: exactly 480 samples at 160 Hz |
| Decoder A | Four CSP projections, covariance regularization 0.1, log mean-square power, shrinkage LDA |
| Decoder B | Per-channel log mean-square bandpower, identical shrinkage LDA |
| LDA | `solver="lsqr"`, `shrinkage="auto"`, equal class priors |

Both decoders receive the same epoch arrays and labels. Filtering is forward-only and independent for each recording. There is no resampling, baseline subtraction, artifact rejection, channel repair, test normalization, adaptation, or hyperparameter search. Run 12 is never used for fitting or selecting parameters.

The shared power transform uses the natural logarithm and a fixed `1e-24` numerical floor in squared input units. CSP projection units differ from sensor volts; the floor is numerical protection, not an artifact threshold. “Original signal” does not mean artifact-free.

## Reproduce

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
uv run bci-stress fetch
uv run pytest -q
uv run bci-stress run --config configs/clean.json --output runs/clean-sub001
```

`fetch` retrieves six files (three SET and three EDF) from OpenNeuro and verifies SHA256 and sizes pinned to dataset commit `af0c461f23e8a7aa782475e3c22d160bc39eb170`. Existing cache files are verified, never silently replaced. No EEG recordings are committed to this repository. After dependencies and data are cached, use `uv run --offline ...`; training and tests need no network. The output directory must be new, preventing accidental overwriting of results.

The loader intentionally supports only the audited participant and run layout. Extending it to other participants requires another audit, not silently assuming every recording has the same timing.

## Checks and artifacts

- Source checksums, channel order, units, sample timing, original event labels/durations, and all 45 epoch boundaries are verified before fitting.
- Train/test run IDs, trial IDs, and signal hashes are checked for overlap. Arrays are read-only; their hashes are checked after all fits.
- Prediction must preserve learned parameters and give the same probabilities whether test epochs are evaluated individually or together.
- Tests cover causal filtering, run boundaries, feature definitions, fit isolation, label shuffling, metric axes, and incomplete-run reporting. Cached real-data tests skip when recordings are absent.
- **199 label permutations**, seeded with `20261005`, shuffle training labels independently within runs 4 and 8. Both entire pipelines are refitted from scratch on the same shuffled labels; run-12 signals and labels stay fixed. This is a diagnostic null reference. Its empirical tail fraction is not a calibrated significance test because temporal exchangeability has not been established. Near-chance permutations do not prove that all leakage is absent.

Every run writes:

| Artifact | Contents |
| --- | --- |
| `config.json`, `protocol.json` | Resolved configuration, explicit feature/classifier parameters, split and preprocessing rules |
| `metrics.json`, `confusion_matrices.json` | Accuracy, balanced accuracy, class recall, confusion counts (rows true, columns predicted; left/right order) |
| `predictions.csv`, `split.json` | Per-trial test predictions/probabilities and exact training/test trial identities |
| `label_permutations.json` | Every shuffled training-label vector and both null scores, seed and summaries |
| `data_audit.json`, `dataset_manifest.json` | Verified input checksums, events, units, channels, filter coefficients and epoch bounds |
| `leakage_checks.json` | Split, input immutability, shared-input fingerprints and prediction-state checks |
| `*_parameters.npz` | Learned LDA parameters and CSP spatial filters/patterns |
| `metadata.json`, `source/` | Code commit and dirty status, source hashes and copies, Python/package/platform versions, thread limit |
| `status.json` | Running, complete, or failed status with timestamps |

Use `metrics.json` only from a run marked complete. The full code/configuration is frozen before the fixed evaluation; permutation scores are diagnostic and do not select a model or alter settings. This single participant with 15 test trials cannot support a population-level robustness claim.

## Audit findings and references

The SET files omit event durations. Source EDF annotations establish 4.1-second imagery intervals and an unannotated final 0.5 seconds; all selected windows fit. SET reference `common` does not establish common-average referencing. All three files indicate session 1, so this evaluates later-run transfer, not another day.

- [Actual-file audit](reports/sub-001-audit.md)
- [Decoder-control rationale](reports/decoder-controls.md)
- [Dataset repository](https://github.com/OpenNeuroDatasets/ds004362)
- [Original data and study citation](https://physionet.org/content/eegmmidb/1.0.0/)
- [CSP implementation](https://mne.tools/stable/generated/mne.decoding.CSP.html)

The OpenNeuro dataset metadata declares CC0. Cite the dataset and original BCI2000 publication when using its recordings. [PLAN.md](PLAN.md) records future work; [prompts.md](prompts.md) records the project requests.
