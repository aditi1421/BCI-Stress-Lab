# Fixed clean-decoding result: sub-001

Evaluated 2026-10-05 using implementation commit `35816797d769a6afb4912c6ee44364556f9964b3`, with a clean working tree and the committed configuration. No settings were selected or changed using run-12 performance.

Training: runs 4+8, 30 epochs (16 left, 14 right). Test: run 12, 15 epochs (7 left, 8 right). Both decoders received the identical 64-channel, 480-sample epochs, after independent causal 8–30 Hz filtering of each continuous recording.

| Decoder | Accuracy | Balanced accuracy | Left recall | Right recall |
| --- | --- | --- | --- | --- |
| CSP + shrinkage LDA | 60.00% (9/15) | 60.71% | 71.43% | 50.00% |
| Per-channel log-bandpower + shrinkage LDA | 46.67% (7/15) | 46.43% | 42.86% | 50.00% |

Confusion matrices use rows=true and columns=predicted, in left/right order:

```text
CSP + LDA          Bandpower + LDA
         L  R               L  R
true L   5  2      true L    3  4
true R   4  4      true R    4  4
```

## Permutation diagnostic

For each of 199 seeded iterations, training labels were shuffled separately within runs 4 and 8; fresh copies of both complete pipelines were fitted using the same shuffled labels. Test signals and labels stayed fixed.

| Decoder | Null mean balanced accuracy | Central 95% null quantiles | Diagnostic tail fraction, plus-one |
| --- | --- | --- | --- |
| CSP + LDA | 50.71% | 27.68%–72.32% | 0.195 |
| Bandpower + LDA | 48.38% | 25.85%–67.05% | 0.585 |

These are diagnostic reference distributions, not calibrated significance tests. Both observed scores fall within their respective central null ranges. Temporal exchangeability is not established. No model was selected based on these results.

The bandpower control is near chance on these 15 test trials. It has little established useful performance to lose, so a later flat corruption curve would not demonstrate practical robustness. The modest CSP result and single-participant sample also do not establish dependable decoding or a population effect. This clean stage makes no claim about corruption tolerance.

## Validation and reproducibility

- All 41 tests passed, including cached real-data validation, source units/events, causal filtering, exact feature calculations, estimator isolation, permutation handling, and incomplete-run behavior. Ruff and whitespace checks passed.
- Source file hashes match the audited dataset commit. All 45 imagery windows fit source-annotated intervals; no trials were rejected.
- Training/test run and trial identities are disjoint; no signal epoch is duplicated across splits. Both pipelines received identical input fingerprints.
- Learned parameters remain unchanged during prediction. Batch and individual predictions agree. Epoch arrays and labels are unchanged after observed and permuted fits.
- The frozen experiment was repeated once solely to check determinism. Metrics, predictions, all 199 paired permutation scores, audits, split checks, fitted parameter arrays, and provenance match exactly. Only timestamps differ. There was no intervening code/configuration change or tuning.

This directory contains all result tables, predictions, configuration, parameter arrays, and reproducibility metadata. The implementation is recoverable at the recorded Git commit. Local runs additionally preserve source-file copies under `runs/clean-sub001/source/`; these duplicate source copies are not committed here. Raw EEG is downloaded and checksum-verified separately.

Reproduce from the repository root:

```sh
uv sync --locked
uv run bci-stress fetch
uv run --offline bci-stress run --config reports/clean-sub001/config.json --output runs/reproduced-clean-sub001
```
