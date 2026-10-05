# BCI Stress Lab — user prompts

Prompts are recorded verbatim in conversation order. Dates use Asia/Kolkata. This log covers the BCI project prompts, not workspace setup instructions.

## 2026-10-05 — Prompt 1

I want to build a BCI Stress lab, which is how How quickly does a motor-imagery BCI break when EEG becomes messy like it would outside a lab? Before writing any code, propose the smallest architecture that could step us towards this goal. I want Python, a public dataset, reproducible experiments, tests, and a simple dashboard. Prioritize correctness over sophistication. Explain the tradeoffs in dataset, preprocessing, and decoder choice.

## 2026-10-05 — Prompt 2

I want to build a BCI Stress lab, which is how How quickly does a motor-imagery BCI break when EEG becomes messy like it would outside a lab? Before writing any code, propose the smallest architecture that could step us towards this goal. I want Python, a public dataset, reproducible experiments, tests, and a simple dashboard. Prioritize correctness over sophistication. Explain the tradeoffs in dataset, preprocessing, and decoder choice.Ask me any clarifying questions. And save these prompts that I send to you as well.

## 2026-10-05 — Prompt 3

Make a plan, we can find out failure modes while going through the training, Offline cpu only for now. Quickly means corruption severity and elapsed time

## 2026-10-05 — Prompt 4

Use this dataset - https://github.com/OpenNeuroDatasets/ds004362.git

## 2026-10-05 — Prompt 5

yes works

## 2026-10-05 — User shell command

```sh
open bci-stress-lab/Plan.md
```

## 2026-10-05 — Prompt 6

Before implementing anything, inspect sub-001 runs 4, 8 and 12 from ds004362. Tell me exactly what the .set files contain: sampling rate, channel names, signal units, event labels, event timings, recording duration, and whether the expected imagery trials are actually present. Flag anything that contradicts our plan

## 2026-10-05 — Prompt 7

We are deliberately corrupting channels, but CSP itself depends heavily on channel covariance. Are we measuring realistic BCI fragility, or mainly measuring a known weakness of CSP? What control experiment would separate those two explanations?

## 2026-10-05 — Prompt 8

Run parallel agents and do the work.

## 2026-10-05 — Prompt 9

Implement the clean decoding stage only. Using the audited runs, extract left/right motor-imagery epochs from [cue+1, cue+4) seconds. Use runs 4 and 8 for training and run 12 as the untouched later-run test set. Implement two pipelines: (1) CSP + shrinkage LDA and (2) per-channel log-bandpower + the identical shrinkage LDA. Keep preprocessing, channels, epochs, and splits identical. Add label-permutation and leakage checks. Save metrics, confusion matrices, exact configuration, and reproducibility metadata. Do not implement signal corruptions yet and do not tune using run 12.

## 2026-10-05 — Prompt 10

https://github.com/aditi1421/BCI-Stress-Lab.git -update here, do clean commits to this github link.

## 2026-10-05 — Prompt 11

default to e he previous model then..

## 2026-10-05 — Prompt 12

now try

## 2026-10-05 — Prompt 13

decoder-controls.md still says “no decoder trained or implementation written”, while the README and clean-result report clearly show that this stage is now complete. Can u check this one?
