---
tags: [related, reference]
---

# ShortcutFM

Sister project by the same author. Flow matching language model with SAE analysis. Separate repo but inspired some patterns here.

## Where

Referenced in workspace rules as `slurm:/home/inf148234/projects/ShortcutFM` (different cluster). Git remote: `git@github.com:jwarczynski/ShortcutFM`.

## Patterns borrowed

- **Single accumulated wandb table for generated samples**. Instead of logging a separate table per epoch (cluttering the wandb UI), accumulate rows into one list and call `logger.log_table("val/generated_samples", columns, data)` every epoch. See `ShortcutFM/shortcutfm/train/pl/train_unit.py::_process_train_batch_predictions`.

  We adopted this in commit ``5b182e9`` for our PlaidXSumTrainingModule — `self._val_predictions` list accumulates across epochs, single `val/generated_samples` table.

- **Pre-tool-use hook pattern**. Not currently used in GenieSAE but useful for sandboxing.

## Not a code dependency

GenieSAE doesn't import from ShortcutFM. Reference only.
