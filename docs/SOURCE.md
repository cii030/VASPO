# Source provenance and review scope

This repository was prepared from the paper project's `code/vaspo/` submission
source on 2026-10-15. The author's separate `critic_init/` source was subsequently
added for offline critic pretraining. The surrounding paper repository, its Git history, local
snapshot metadata, submission archives, datasets, and checkpoints are not part
of this import.

## Implementation retained

- `verl/` contains the supplied modified training framework, including rollout
  implementations and Hydra configuration files.
- `rlcsd/` contains the privileged teacher integration, baseline interfaces,
  reward adapter, and VASPO diagnostics. Its package name is preserved.
- The original CPU tests are retained. `scripts/run_vaspo.sh` now forwards to
  the Qwen3 mathematics example in `scripts_run/`.
- Upstream packaging metadata, dependencies, licenses, and source headers are
  preserved. Installing this project provides the `verl` distribution.

The initial import did not change the model, loss functions, training schedule,
or hyperparameters. Subsequent launch examples align settings with the paper,
including 50-step warmup and sampled validation; see `scripts_run/README.md`.
Documentation and ignore rules are adapted for an independent
GitHub repository. The source archive/audit scripts now retain nested rollout
implementation directories and write the archive under `dist/`.

## Validation records

`submission-validation.json` and `submission-privacy-audit.json` are unchanged
records from the submission preparation. Their reported environment and test
counts describe that earlier snapshot. Current tests and source audits must be
run independently; the historical records do not certify this checkout.

## Critic pretraining import

`critic_init/` comes from the author's value-model project. Its Python training,
loss, response-mask, checkpoint repair, and data-conversion files are retained
without changes. Bytecode caches are excluded. The launcher replaces local
cluster paths with environment variables and uses the included ZeRO-1 config.
Its defaults follow the paper's critic record: two epochs, learning rate 5e-6,
256-step accumulation, bfloat16, and eight GPUs. Dependencies and usage are
documented in that directory. The JSONL examples illustrate the schema and are
not the full pretraining corpus.

## Reproduction assets

The repository now includes offline critic conversion and pretraining. Initial
actor trajectories and verifier labels, final benchmark evaluation scripts,
full datasets, trained weights, and exact historical run configurations still
need to be supplied. The training launcher supplies starting settings documented
in the source README. Full distributed GPU training must be validated on the
intended training hardware with those assets.
