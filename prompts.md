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

## 2026-10-05 — Prompt 14

Use multiple agents in parallel. The goal of this phase is to run the first real robustness experiment:

How does motor-imagery decoding performance degrade as EEG channels progressively fail or the signal is corrupted, and do CSP + LDA and channel-bandpower + LDA fail differently?

Do not broaden the scope beyond this question yet.

Agent roles

Agent 1 — Experiment Design / Research Lead

Review the current repository, PLAN.md, audit reports, decoder controls, and clean baseline results.

Your job is to:

verify the existing experimental protocol

define the corruption severity grid before results are inspected

ensure train/test separation remains intact

decide exactly how Gaussian noise and channel flatlining should be applied

define the metrics and saved outputs

identify any methodological issue that could invalidate interpretation

Important constraints:

runs 4 + 8 remain training

run 12 remains held-out later-run testing

no tuning on run 12

corrupt the continuous test EEG before filtering/epoch extraction

use the same preprocessing for both decoders

keep the trained decoders frozen during the primary stress test

this is still a single-subject pilot

do not claim cross-day robustness

synthetic corruption is not equivalent to real-world artifact realism

Write the finalized protocol into a concise experiment config/document before execution.

Agent 2 — Corruption Infrastructure

Implement deterministic EEG corruption operators.

Start with only:

Additive Gaussian noise

Channel flatlining/dropout

Requirements:

operate on continuous EEG

preserve shape

never mutate input arrays

fixed/random-state-controlled seeds

zero corruption must return numerically identical data

corruption should operate in the dataset's native signal units

make severity definitions explicit and interpretable

For Gaussian noise, use severity relative to a clearly defined signal scale such as training-data RMS. Do not choose arbitrary absolute amplitudes without documenting why.

For channel flatlining:

support random subsets of channels

support explicit channel names

save exactly which channels were removed for each run

keep random channel selection reproducible

Add comprehensive tests.

Do not run the full experiment until the testing agent has reviewed this code.

Agent 3 — Validation / Adversarial Reviewer

Act as a skeptical reviewer.

Review:

corruption implementation

preprocessing order

units

epoch timing

leakage risks

frozen-model behavior

seed handling

metric computation

Try to find bugs or experimental design flaws.

Specifically test:

zero corruption reproduces the clean baseline

repeated runs with the same seed produce identical outputs

source data are unchanged after corruption

training data are not accidentally corrupted during frozen-decoder tests

corruption occurs before filtering

both decoder pipelines see equivalent corrupted recordings

run 12 is never used for parameter selection

chance-level controls are not described as robust simply because their accuracy curve is flat

Report any issue before the main experiment proceeds.

Agent 4 — Experiment Runner / Analysis

Only after Agents 1–3 agree the implementation is valid, run the first stress experiment on sub-001.

Use the already-fixed decoders:

CSP + shrinkage LDA

per-channel log-bandpower + shrinkage LDA

First experiment:

Gaussian noise

Pre-register/freeze a severity grid such as:

0
0.25 × signal RMS
0.5 × signal RMS
1.0 × signal RMS
2.0 × signal RMS

Adjust only if the research-design agent identifies a principled reason.

Channel failure

Use:

0
1
2
4
8
16 failed channels

For random channel failure:

use multiple fixed seeds per severity

report mean and spread

save which channels were failed for every replicate

For every condition save:

decoder

corruption type

severity

seed

failed channel names if applicable

accuracy

balanced accuracy

delta balanced accuracy relative to the same decoder's clean baseline

predictions

confusion matrix

exact experiment configuration

Produce machine-readable CSV/JSON results.

Agent 5 — Visualization / Interpretation

Using only the frozen experiment outputs, generate:

Balanced accuracy vs Gaussian-noise severity

Delta balanced accuracy vs Gaussian-noise severity

Balanced accuracy vs number of failed channels

Delta balanced accuracy vs number of failed channels

Show both decoders on the same comparisons where appropriate.

Do not over-interpret the curves.

Explicitly distinguish:

absolute decoding performance

degradation relative to clean baseline

This distinction matters because the bandpower control currently has weak clean performance. A flat near-chance curve is not evidence of robustness.

Write a short results report answering:

Does CSP performance appear sensitive to either corruption?

Does bandpower behave differently?

Are any apparent differences meaningful given only 15 held-out trials?

What conclusions cannot be made yet?

What experiment should logically come next?

Do not describe this as general BCI fragility.

Integration Agent / Lead Agent

After all agents finish:

Review every contribution.

Resolve disagreements using the most conservative scientifically defensible interpretation.

Run the full test suite.

Reproduce the zero-corruption clean baselines.

Run the finalized experiment.

Update the repo documentation with the new experiment and results.

Save results under a clearly named report directory.

Keep all configs/seeds required for reproduction.

Make clean, scoped commits rather than one large commit.

Suggested commit structure:

feat: add deterministic EEG corruption operators

test: validate corruption and frozen-decoder pipeline

feat: add frozen decoder stress evaluation

analysis: add sub-001 corruption experiment results

docs: document first BCI stress-test findings

Do not silently modify the clean baseline protocol.

Do not optimize model performance during this phase.

Do not add additional corruption families yet.

Do not make population-level conclusions from sub-001.

At the end, provide me with:

the exact experiment that was run

the resulting metrics

the plots generated

any methodological concerns found by the reviewer agent

what changed in the repo

the commit hashes

your recommendation for the single most informative next experiment
