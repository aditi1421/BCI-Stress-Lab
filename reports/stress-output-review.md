# Independent review of the frozen stress outputs

Disposition: **passed; no output-consistency blocker found**. This review inspected the completed artifacts in `reports/stress-sub001`. It did not rerun models, score additional conditions, change parameters, or modify the frozen pre-execution protocol/review.

## Checks performed

- Recomputed accuracy, class recalls, balanced accuracy and every confusion matrix directly from all **3,300 trial predictions**. All **220 decoder-condition rows** agree exactly, including prediction/probability class consistency and constant-prediction flags.
- Verified every condition retains the same 15 run-12 trial identities, labels and 480-sample windows. Both decoders share each condition's epoch fingerprint. The full grid contains exactly **110 conditions** with the prescribed severities and seeds.
- Independently recomputed baseline-relative deltas and all **22 severity/decoder summaries**: means, sample standard deviations (`ddof=1`), extrema and constant-prediction counts. JSON and CSV metrics/summaries agree exactly.
- All **20 zero-corruption conditions** retain the original continuous and epoch fingerprints, predictions, probabilities, confusion matrices and metrics. Their repeated seed rows are duplicate references, not new observations.
- Every flatline mask matches the corresponding seeded permutation prefix; channel names, indices and counts agree. Masks are nested within seed. Gaussian requested standard deviations equal severity times saved training RMS, and zero-noise realized RMS is zero.
- Independently recomputed the 64 training RMS values from the raw cached runs 4 and 8; all match exactly. No test samples enter that calculation.
- Recomputed model fingerprints from the committed clean parameter arrays. Every condition's before/after fingerprints match those originals. Saved input fingerprints remain unchanged, and all source dataset file sizes/checksums match the pinned manifest.
- Clean configuration and metrics match the original baseline. Stress configuration matches the frozen grid. Copied protocol and pre-execution-review file hashes match the execution metadata.

## Interpretation requiring emphasis

At Gaussian severities **0.5, 1 and 2**, both decoders predict one class for every trial in every seed: CSP always predicts **left**, while channel-bandpower always predicts **right**. Each therefore obtains 50% balanced accuracy despite complete loss of class discrimination in its predictions. The corresponding ordinary accuracies differ (7/15 versus 8/15) solely because test class counts differ.

Channel-bandpower produces constant predictions in **48 of 50 nonzero flatline conditions**, including all ten one-channel-failure replicates. Its mean balanced accuracy of approximately 50% is not robustness. Its positive baseline-relative change is largely a consequence of the original 46.43% baseline being below 50%. Whole-recording flatlines drive failed-channel power to the existing numerical floor, creating extreme feature extrapolation for the frozen classifier.

CSP's mean balanced accuracy decreases from 60.71% to 57.77% at 0.25× RMS Gaussian noise, and to 50% at the higher sampled noise levels. Under flatlining, means fall to 57.77%, 55.27%, 53.93%, 54.46% and 54.29% for 1, 2, 4, 8 and 16 failed channels. This is sensitivity under the specified experiment; it is neither monotonic for every severity nor proof of a general CSP-specific mechanism.

The channel-bandpower mean increase to 58.21% at 0.25× RMS noise should remain an observed, unconfirmed fluctuation/change in this fixed trial set. It does not establish noise as a useful intervention. Neither decoder's clean result established dependable above-chance performance against its permutation reference.

Only 15 biological test trials and one subject support these curves. Seed spread describes conditional corruption variability and cannot justify population confidence intervals, significance tests treating seeds as independent participants, or claims about real-world artifact realism. This experiment also supplies no elapsed-time-to-failure estimate.

The most informative next mechanism check remains a separately frozen, training-only **known-fault refit** comparison, preferably evaluated on fresh held-out recordings. It can test whether matched calibration recovers performance without claiming that persistent failure proves irrecoverable information loss.
