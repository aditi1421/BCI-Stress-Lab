# Separating CSP sensitivity from broader decoder fragility

Status: the clean decoding stage is implemented and evaluated. The [actual-file audit](sub-001-audit.md) verified all selected trial windows. Both decoders were trained on runs **4 and 8** and evaluated on the fixed later-run test set, **run 12**, without test-set tuning. The [completed result report](clean-sub001/README.md) records 60.71% balanced accuracy for CSP + LDA and 46.43% for per-channel log-bandpower + LDA, together with 199 label permutations, leakage checks, and an exact reproducibility repeat.

The [first frozen-decoder corruption experiment](stress-sub001/README.md) is now complete for Gaussian noise and random channel flatlining, with the original clean models verified and unchanged. Known-fault refits, exhaustive single-channel location sweeps, and elapsed-time experiments remain future work. The sections below retain the design rationale; the [frozen stress protocol](stress-protocol.md) specifies the executed grid and delta convention.

## What the original experiment can establish

Both explanations can contribute. A channel fault can remove useful motor information and can make a calibrated decoder's features behave differently from training. A CSP-only experiment establishes robustness of that particular pipeline under the stated perturbations, not universal BCI robustness or the realism/frequency of those perturbations.

There is also an important distinction: with test-only corruption and a **frozen CSP**, its training covariance matrices and spatial filters are not re-estimated. Thus failure is not directly evidence of corrupted covariance *estimation during fitting*. The fixed projection receives altered spatial covariance; its projected powers and the LDA decision inputs change. The original CSP paper distinguishes estimation sensitivity from post-fit artifact effects. [Ramoser et al., 2000, discussion](https://www.cs.hmc.edu/~keller/eeg/Ramoser.pdf)

## Implemented paired decoders

| Arm | Features | Classifier |
| --- | --- | --- |
| A: current baseline | Four regularized CSP projections, followed by log mean-square power | Shrinkage LDA |
| B: spatial-filter control | One log mean-square bandpower feature per input EEG channel; no learned spatial mixing | Same shrinkage LDA settings |

The clean implementation uses identical EEG channels, recorded reference, causal temporal filtering, eligible trial windows, and train/test splits for both arms, with no trial rejection or interpolation. Both use a fixed numerical power floor of `1e-24` in squared input units (sensor volts or CSP projection units); it was set before test evaluation.

The first corruption comparison uses identical corrupted signal copies, channel masks, amplitude grids, and seeds for both arms. Gaussian scaling uses training recordings only. The full grid completed without numerical failures; no conditions were excluded.

CSP is a supervised spatial decomposition whose covariance estimation and power transform are configurable. Arm B removes this learned spatial decomposition. However, **B is not covariance-free**: LDA still estimates feature covariance. Its feature dimension also differs from A. The comparison tests practical sensitivity associated with the CSP feature pipeline, not a pure causal attribution to one mathematical operation. [MNE CSP documentation](https://mne.tools/stable/generated/mne.decoding.CSP.html), [scikit-learn LDA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

The original proposal described positive loss as `drop_d(s) = BA_d(original) - BA_d(s)`. Executed stress artifacts instead use **`delta_balanced_accuracy = BA_d(s) - BA_d(original)`**, so negative values mean degradation. Show absolute balanced accuracy alongside these paired baseline-relative changes. Larger CSP degradation can support additional sensitivity of that pipeline, subject to the different baseline performance; similar degradation can support a shared sensitivity among these two decoders. Neither result proves universal BCI fragility.

Always display original baseline performance. A control already near chance has little useful performance to lose; its flatter curve is not evidence of greater practical robustness. Do not select participants using test accuracy or degrade CSP deliberately to equalize baselines. Treat inconclusive low-baseline cases as findings. Across participants, aggregate paired within-participant effects; corruption seeds and overlapping time windows are not independent participants. One participant is a mechanism pilot, not a population result.

## Two small follow-up diagnostics

1. **Known-fault refit:** At selected severities, train fresh copies of both arms on independently corrupted *training* data with the same fault type/severity/channel mask as testing. Do not fit on test labels or test covariance. Also evaluate each refitted decoder on the original test signal. Recovery suggests a train/test mismatch that calibration could address. Persistent failure is compatible with lost usable information or model limitations; it does not prove information-theoretic impossibility. Call this a known-condition adaptation diagnostic, not a guaranteed upper bound. Keep it separate from frozen-decoder results.
2. **Channel-location control:** For single-channel faults, examine every channel separately using paired trials. Report motor-area versus other locations using an anatomical grouping fixed before scoring, not rankings learned from test accuracy. Then add uniformly sampled nested masks for multi-channel loss. This distinguishes fault count from *which* information was disturbed and avoids making “random dropout” depend on a lucky mask. A deliberately targeted high-CSP-weight fault is an additional worst-case analysis, not a typical-fault estimate.

If closer dimensional matching becomes necessary, add a prespecified fixed spatial-filter bank with the same number of outputs. That is a later attribution check, not required for the smallest useful two-arm experiment.

## Severity and elapsed time

Run both arms on the same fixed-severity step exposures, with multiple label-independent onset schedules. Show elapsed recording time, severity, trial exposure, and class counts. Analyze onset/filter recovery separately from steady exposure. A frozen, trial-wise decoder does not itself accumulate experience or fatigue; changes over time can reflect filter transients, which trials occur, or the underlying EEG. A single increasing ramp cannot separate elapsed time from severity. Short cued recordings and sparse trials may leave too few observations for a stable time-to-failure estimate.

## Recommendation

Keep **CSP + LDA versus per-channel bandpower + LDA** as the paired comparison. The clean bandpower control is near chance on the 15 held-out trials and both original scores fall within their respective permutation reference ranges. In the first stress experiment, every one-channel flatline replicate makes bandpower predict a single class, despite its apparently improved 50% balanced accuracy. Both decoders become constant-class predictors at Gaussian severity 0.5× training RMS and above. These outcomes show sensitivity of the specific frozen pipelines; they do not establish useful robustness or justify tuning against run 12.

Known-fault refitting and single-channel location experiments remain follow-up diagnostics. Any future stress results should be described as controlled robustness of these decoders on recorded EEG. Realistic outside-lab fragility will need empirically grounded artifacts or recordings collected under those conditions; swapping classifiers alone cannot establish it.
