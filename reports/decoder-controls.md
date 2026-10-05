# Separating CSP sensitivity from broader decoder fragility

Status: proposed control design; no decoder trained or implementation written. Dataset and trial-window eligibility remain subject to the independent `.set` audit. Existing split: run 4 development training, run 8 development evaluation, run 12 final evaluation after protocol freeze and refitting on 4+8.

## What the original experiment can establish

Both explanations can contribute. A channel fault can remove useful motor information and can make a calibrated decoder's features behave differently from training. A CSP-only experiment establishes robustness of that particular pipeline under the stated perturbations, not universal BCI robustness or the realism/frequency of those perturbations.

There is also an important distinction: with test-only corruption and a **frozen CSP**, its training covariance matrices and spatial filters are not re-estimated. Thus failure is not directly evidence of corrupted covariance *estimation during fitting*. The fixed projection receives altered spatial covariance; its projected powers and the LDA decision inputs change. The original CSP paper distinguishes estimation sensitivity from post-fit artifact effects. [Ramoser et al., 2000, discussion](https://www.cs.hmc.edu/~keller/eeg/Ramoser.pdf)

## Minimum useful paired control

| Arm | Features | Classifier |
| --- | --- | --- |
| A: current baseline | Four regularized CSP projections, followed by log mean-square power | Shrinkage LDA |
| B: spatial-filter control | One log mean-square bandpower feature per input EEG channel; no learned spatial mixing | Same shrinkage LDA settings |

Use the exact same EEG channels, recorded reference, causal temporal filter, eligible trial windows, split, corrupted signal copies, channel masks, amplitude grid, and seeds for both arms. No trial rejection or interpolation in either arm. Any feature scaling and the fixed floor required to keep log power finite are set from training data or a preregistered physical-unit constant, never from test trials. Save numerical failures explicitly.

CSP is a supervised spatial decomposition whose covariance estimation and power transform are configurable. Arm B removes this learned spatial decomposition. However, **B is not covariance-free**: LDA still estimates feature covariance. Its feature dimension also differs from A. The comparison tests practical sensitivity associated with the CSP feature pipeline, not a pure causal attribution to one mathematical operation. [MNE CSP documentation](https://mne.tools/stable/generated/mne.decoding.CSP.html), [scikit-learn LDA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

For decoder d and severity s, compute `drop_d(s) = BA_d(original) - BA_d(s)`. Compare the paired difference `drop_CSP(s) - drop_bandpower(s)` and show both absolute balanced-accuracy curves. Larger CSP degradation supports an additional CSP-pipeline vulnerability; similar degradation supports a shared vulnerability among these two decoders. Neither result proves universal BCI fragility.

Always display original baseline performance. A control already near chance has little useful performance to lose; its flatter curve is not evidence of greater practical robustness. Do not select participants using test accuracy or degrade CSP deliberately to equalize baselines. Treat inconclusive low-baseline cases as findings. Across participants, aggregate paired within-participant effects; corruption seeds and overlapping time windows are not independent participants. One participant is a mechanism pilot, not a population result.

## Two small follow-up diagnostics

1. **Known-fault refit:** At selected severities, train fresh copies of both arms on independently corrupted *training* data with the same fault type/severity/channel mask as testing. Do not fit on test labels or test covariance. Also evaluate each refitted decoder on the original test signal. Recovery suggests a train/test mismatch that calibration could address. Persistent failure is compatible with lost usable information or model limitations; it does not prove information-theoretic impossibility. Call this a known-condition adaptation diagnostic, not a guaranteed upper bound. Keep it separate from frozen-decoder results.
2. **Channel-location control:** For single-channel faults, examine every channel separately using paired trials. Report motor-area versus other locations using an anatomical grouping fixed before scoring, not rankings learned from test accuracy. Then add uniformly sampled nested masks for multi-channel loss. This distinguishes fault count from *which* information was disturbed and avoids making “random dropout” depend on a lucky mask. A deliberately targeted high-CSP-weight fault is an additional worst-case analysis, not a typical-fault estimate.

If closer dimensional matching becomes necessary, add a prespecified fixed spatial-filter bank with the same number of outputs. That is a later attribution check, not required for the smallest useful two-arm experiment.

## Severity and elapsed time

Run both arms on the same fixed-severity step exposures, with multiple label-independent onset schedules. Show elapsed recording time, severity, trial exposure, and class counts. Analyze onset/filter recovery separately from steady exposure. A frozen, trial-wise decoder does not itself accumulate experience or fatigue; changes over time can reflect filter transients, which trials occur, or the underlying EEG. A single increasing ramp cannot separate elapsed time from severity. Short cued recordings and sparse trials may leave too few observations for a stable time-to-failure estimate.

## Recommendation

Make **CSP + LDA versus per-channel bandpower + LDA** mandatory in the first implementation. Add known-fault refitting and single-channel location results only after the two baselines are usable. Retain the claim “controlled robustness of these decoders on recorded EEG.” Realistic outside-lab fragility will later need empirically grounded artifacts or recordings collected under those conditions; swapping classifiers alone cannot establish it.
