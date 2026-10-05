# First stress experiment: adversarial review

Status: **approved for the frozen protocol**, before any nonzero real-data stress result was inspected. No implementation defect or design blocker remains in the reviewed paths. The interpretation limitations below remain material.

## Scope and initial findings

The existing clean stage was independently reproduced from the cached, checksum-verified recordings before reviewing stress results. The four train/test array and label fingerprints, every stored learned parameter array, and the complete clean metric dictionaries matched exactly. CSP balanced accuracy is 60.7143%; channel-bandpower balanced accuracy is 46.4286%.

The fixed split remains runs 4+8 for training and run 12 for later-run testing. The existing preprocessing is a forward-only fourth-order Butterworth bandpass prototype (8–30 Hz; eighth-order realized bandpass), zero filter state reset per recording, preserved reference, all 64 channels, and half-open `[cue+1, cue+4)` epochs with exactly 480 samples. Neither clean protocol nor fitted decoder settings may change during this experiment.

## Execution gate: passed

- Native microvolt corruption precedes conversion to volts, the shared continuous filter, and epoch extraction. Synthetic instrumentation verifies the exact complete recording passed to the filter for both corruption families.
- Noise scale uses raw, unfiltered training recordings only. An adversarial synthetic test increases run-12 amplitude by `1e8` while leaving the scale exactly unchanged. The real-data scale independently matches pooled training RMS over 40,000 samples per channel.
- Noise draws and nested channel masks repeat exactly for a fixed seed. Source arrays, training epochs, labels and scales retain their fingerprints; source file checksums also remain unchanged.
- Both decoders receive the identical corrupted epoch array object. Every estimator `fit` method is replaced by a failing trap during test predictions; all checks pass. Learned-array fingerprints remain unchanged.
- Both corruption families at zero, with seeds 0 and 9, reproduce the original real-data clean epochs, predictions, probabilities, metrics and confusion matrices. No nonzero real-data conditions were evaluated during this gate.
- Exact clean configuration and saved parameter-array equality prevent silent protocol changes or model adaptation. A changed CSP component setting is rejected before source loading.
- Independent metric recomputation and summary tests confirm confusion axes, delta sign (`stressed - clean`), sample standard deviation (`ddof=1`), constant-class detection and duplicate-seed rejection.

Validation command: `.venv/bin/pytest -q tests/test_stress_review.py tests/test_corruption.py` — **70 passed** (12 reviewer checks and 58 corruption checks). `.venv/bin/ruff check tests/test_stress_review.py` also passed. The integration lead must run the full repository suite before executing the finalized grid.

Integration gate: the full repository suite passed **116 tests**, with Ruff and whitespace checks passing before primary execution. The lead also found and corrected a plotting-only floating-point edge case: the mean of identical scores may round one unit outside their min/max. A `1e-12` validation tolerance and regression test prevent rejecting valid summaries. This did not change any experiment parameter, metric, or clean decoder.

## Interpretation constraints

1. Raw training RMS includes recorded offsets, rest, and any pre-existing artifacts. Gaussian severity is relative to that explicit scale, not to clean neural power or an in-band signal-to-noise ratio. Temporal filtering removes part of the injected broadband noise.
2. A whole-recording zero flatline is a specific synthetic failure. It is not interpolation, removal of a feature dimension, intermittent contact loss, or a realistic electrode impedance model. With zero filter state, its bandpower is exactly zero and its log feature reaches the pre-existing `1e-24` power floor. This is extreme feature extrapolation for the frozen channel-bandpower classifier.
3. A constant-class classifier scores 50% balanced accuracy here. The weak channel-bandpower baseline can therefore show an apparent improvement when useful discrimination collapses. Absolute scores, class recalls/confusion matrices, and baseline-relative changes must be interpreted together; a flat near-chance curve is not robustness.
4. A 15-trial test set contains 7 left and 8 right trials. One changed left prediction moves balanced accuracy by 7.1429 percentage points; one changed right prediction moves it by 6.25 points. Corruption-seed spread is conditional perturbation variability, not a confidence interval over subjects or future trials.
5. CSP and channel-bandpower differ in feature dimension and spatial mixing; both still use LDA feature covariance. Differences cannot isolate covariance estimation as the sole cause. Frozen CSP does not re-estimate covariance during the stress test.
6. This is one subject, one later run in the same session, with synthetic stationary corruption. It cannot establish general BCI fragility, cross-day reliability, real-world artifact frequency, or time until failure.

## Disposition

Approved to run the grid frozen in `configs/stress.json` and `reports/stress-protocol.md`, with unchanged `configs/clean.json` and the original saved models reconstructed and verified exactly. The reviewed condition loop has no fitting or parameter-selection operation. All nonzero reviewer checks used synthetic signals; only zero-corruption conditions used the actual held-out run. Findings above constrain interpretation even though the implementation passed.
