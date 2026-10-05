# Frozen protocol: sub-001 continuous EEG corruption pilot

Status: design approved before stress scores are generated. The grid in
[`configs/stress.json`](../configs/stress.json) is fixed. Execution requires
the validation review to pass; save this document and its hash with results.
The original-signal baseline has already been observed. This is a prospective
freeze of the corruption analysis, not a preregistration predating that baseline.

## Question and fixed decoding protocol

How do the two fixed decoder pipelines respond to progressively stronger
Gaussian noise or more flatlined EEG channels on the same later-run trials?

Use only audited ds004362 **sub-001**, dataset commit
`af0c461f23e8a7aa782475e3c22d160bc39eb170`. Runs **4+8** remain training
(30 trials: 16 left, 14 right); **run 12** remains held-out testing
(15 trials: 7 left, 8 right). Keep all 64 channels in audited order and the
recorded reference. There is no model selection, adaptation, or test-derived scale.

Retain [`configs/clean.json`](../configs/clean.json) and the clean pipeline:
160 Hz; causal 8–30 Hz Butterworth SOS, prototype order 4; zero filter state
at each run start; no padding, rereferencing, centering, or rejection. Extract
half-open `[cue+1, cue+4)` windows of 480 samples, with the existing 2 s minimum
run-start warmup. Keep the same CSP regularization 0.1 and four components,
log mean-square power with numerical floor `1e-24`, and identical LDA
(`solver="lsqr"`, `shrinkage="auto"`, priors `[0.5, 0.5]`) in both arms.
The control has one log-power feature per channel and no learned spatial mixing.

Reconstruct both original models once from runs 4+8 using the existing clean
`fit_decoders` implementation. Require full configuration equality with the
saved clean configuration and exact equality of every learned array to the
committed `reports/clean-sub001/*_parameters.npz` files. Verify original
predictions and metrics against the saved baseline before nonzero scoring.
Freeze both fitted pipelines before condition evaluation; no condition loop
may call `fit`. Reuse the existing causal filter and epoch extractor. Check learned
parameter fingerprints before and after prediction. Never fit on corrupted
test data, its labels, or its covariance. The original baseline is
60.7142857% balanced accuracy for CSP/LDA and 46.4285714% for bandpower/LDA.

## Corruption definitions

Read a copy of the complete continuous run-12 SET array in its native **µV**
units. Corrupt **before** conversion to volts, causal filtering, and epoch
extraction. Apply a fault throughout the full 125 s recording, including rest.
Then convert to volts exactly as in the clean loader. Each corrupted recording
is filtered and epoched once; both decoders receive the same resulting epochs.
Training recordings are never corrupted. Sources and caller inputs stay unchanged.

| Family | Fixed severities | Replicates |
| --- | --- | --- |
| Additive Gaussian noise, all channels | 0, 0.25, 0.5, 1, 2 × training RMS | Seeds 0–9 |
| Flatline, random channel subsets | 0, 1, 2, 4, 8, 16 channels | Seeds 0–9 |

**Gaussian noise:** for channel `c`, define
`scale[c] = sqrt(sum(x[c, t] ** 2 over runs 4 and 8) / 40000)` in µV.
Use every unfiltered continuous training sample, including rest, with no
mean subtraction. For each seed, draw independent standard-normal values
for all 64 × 20000 channel/sample positions using NumPy `default_rng(seed)`.
At severity `s`, add `s * scale[c] * noise[c, t]`. Reuse that same standard-normal
realization across severities, so only its amplitude changes. Noise is independent
of labels. Save the training RMS vector, requested standard deviations in µV,
and realized injected RMS per channel. The requested standard deviation is an
expected noise RMS, not an assertion of exact finite-sample RMS.

**Flatline:** for each seed, uniformly permute the complete audited channel
list with NumPy `default_rng(seed)`. Flatline the first `k` channels for severity
`k`, yielding nested subsets across severities. Replace every selected sample
by **zero in the recorded reference**, preserving channel positions and count.
Save exact channel names and indices for every replicate. The operator also
supports explicit channel names for reproducible unit checks; this experiment
uses only the specified random subsets. There is no interpolation or repair.

Each family/seed has its own freshly initialized RNG. Numerical seed reuse
across families does not give the families physically paired disturbances.
Zero severity must return a separate, numerically identical copy with no
unit round-trip error. Retain zero rows for every seed as identity checks;
they are duplicate references, not extra observations.

## Validation gate and saved outputs

Before scoring nonzero test conditions, the reviewer must verify:
zero-corruption reproduction of clean epochs/predictions/metrics; seeded
repeatability; input/source immutability; training-only amplitude scales;
corruption before filtering; identical decoder inputs; unchanged fitted model
parameters; correct epoch boundaries/units; split isolation; and metric axes.
No parameter, grid, channel selection, exclusion, or stopping rule may be
changed in response to run-12 scores. Numerical failures remain explicit;
do not omit failed conditions from summaries.

The full grid has **110 recording conditions and 220 decoder-condition rows**.
For every decoder/family/severity/seed, save accuracy, balanced accuracy,
class recalls, left/right confusion counts (rows=true, columns=predicted),
all trial IDs/true labels/predictions/probabilities, and affected channel names.
Define `delta_balanced_accuracy = stressed - original` for the **same decoder**;
negative values mean degradation. Machine-readable values are fractions and
plots/report deltas are percentage points. This sign is the reverse of the
positive-loss `drop` notation in the earlier decoder-control proposal.

Save CSV/JSON result and prediction tables; per-severity mean, sample standard
deviation (`ddof=1`), minimum and maximum over the ten seeds; source hashes and
split; clean and stress configurations; this protocol; native-unit scales and
corruption metadata; filter coefficients; original baseline results; learned
parameter fingerprints; RNG implementation/seeds; Python/package versions;
Git revision, dirty status and source hashes; validation checks; completion status.
Do not recompute label permutations per stress condition: the already-frozen
clean permutation diagnostic supplies baseline context and selects nothing.

Generate four comparisons with both decoders: balanced accuracy and its delta
against Gaussian severity, and balanced accuracy and its delta against failed
channel count. Show mean and seed spread; identify 50% chance and each original
baseline. Save numeric summaries behind every plot. Seed spread is descriptive
conditional on this fixed recording, **not** a confidence interval or population
uncertainty. Do not perform significance tests over seeds as independent subjects.

## Interpretation limits and next experiment

- The control's 46.43% baseline and CSP's 60.71% both fall within their clean
  permutation reference ranges. A flat near-chance curve cannot demonstrate
  useful robustness; report absolute performance beside baseline-relative loss.
- Fifteen test trials produce coarse changes: one left error changes balanced
  accuracy by about 7.14 percentage points; one right error by 6.25 points.
  Ten seeds do not enlarge the participant or biological trial sample.
- Uncentered full-band training RMS includes reference offsets and energy that
  the bandpass removes. The severity is a documented raw-signal amplitude ratio,
  **not in-band SNR**. It is intentionally fixed without test-derived normalization.
- Zero flatlines are a fault proxy. They create log-power-floor features in the
  bandpower arm and change fixed CSP projection powers; this can reflect both
  lost signal and train/test feature mismatch. It does not model amplifier
  saturation, loose electrodes, or empirically validated movement artifacts.
  Inspect saved confusion matrices: 50% balanced accuracy can be constant-class
  collapse, not preservation of decoding ability.
- CSP/LDA and bandpower/LDA differ in feature dimensions and spatial mixing;
  LDA in both still estimates feature covariance. Differences cannot isolate a
  single mathematical mechanism or establish general BCI fragility.
- This is one participant's within-session later-run test, not cross-day or
  population robustness. Full-record exposure measures severity only; it supplies
  no elapsed-time-to-failure or recovery estimate. No failure threshold is inferred.

The most informative next mechanism experiment is a **separately frozen
known-fault training refit** of both arms, using only independently corrupted
runs 4+8 and matched prescribed test conditions. Compare it with the frozen
decoder results to probe calibration mismatch versus lost usable information.
Prespecify conditions without chasing this pilot's largest test effects and
evaluate on fresh held-out data for independent confirmation. Recovery supports
mismatch; persistent failure does not prove irrecoverable information loss.
