---
tags: [infra, exca]
---

# Exca workflows

`exca` wraps Pydantic configs with two capabilities: remote job submission (via submitit) and result caching.

Docs: https://facebookresearch.github.io/exca/infra/introduction.html

## Pattern we use

Every pipeline stage has a Pydantic config class (e.g. `PlaidXSumConfig`, `SAETrainingConfig`) with an `infra: exca.TaskInfra` field and an `@infra.apply`-decorated method called `apply()`.

```python
class MyConfig(BaseModel):
    param_a: int = 10
    infra: exca.TaskInfra = exca.TaskInfra(version="1")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("batch_size",)
    
    @infra.apply
    def apply(self) -> str:
        # do the work
        return "path/to/result"
```

## UID and caching

Exca computes a UID hash from all non-default Pydantic fields **except** those in `_exclude_from_cls_uid`. Same UID → same cache folder → re-runs return instantly.

Cache folder structure:
```
experiments/cache/<folder>/<module>.<ClassName>.<method>,<version>/<uid>/
```

Example:
```
experiments/cache/plaid_xsum_v3b/
    geniesae.configs.plaid_xsum_config.PlaidXSumConfig.apply,1/
        num_gpus=8,seq_len=256,num_epochs=100,...-c8cc5b20/
            logs/
            <result files>
```

## When cache invalidates

- Change any UID-affecting field → new UID → new folder → runs fresh.
- Change `infra.version` → also forces fresh run.
- Bump `version` in the class itself (e.g. `version="2"`) when you change the *code* but not the config.

## CLI submission

```bash
# Submit to slurm
uv run python main.py <subcommand> <config.yaml> --submit --infra.cluster=slurm

# Run inline (no queue, blocks)
uv run python main.py <subcommand> <config.yaml>

# Local subprocess (runs in a child process)
uv run python main.py <subcommand> <config.yaml> --submit --infra.cluster=local

# Override any field
uv run python main.py train-sae configs/... --layer_idx=4 --learning_rate=1e-4

# Job array (when the command supports it)
uv run python main.py train-sae configs/... --layers 0 4 10 14 20 23 --submit --infra.cluster=slurm
```

## Force re-run

```bash
# Option 1: CLI override (preferred)
--infra.mode=force
# or
--infra.force=true

# Option 2: delete cache folder
rm -rf experiments/cache/<folder>/<...>/<uid>/

# Option 3: bump version
# Edit the yaml: infra.version: "2"
```

## Useful infra fields in YAML

```yaml
infra:
  version: "1"
  folder: "./experiments/cache/<task_name>"
  cluster: slurm
  gpus_per_node: 1
  tasks_per_node: 1           # usually =gpus_per_node for DDP
  cpus_per_task: 4
  mem_gb: 64
  timeout_min: 480            # max 2880 on plgrid-gpu-a100
  slurm_partition: plgrid-gpu-a100
  slurm_use_srun: true         # needed for multi-GPU DDP
```

## Debugging a submitted job

```bash
# Get the cache folder for a running job (from submission output)
# "... through cluster 'slurm' (job_id=XXX)"

# Logs are here
find experiments/cache -path "*XXX*log.out" -exec tail -40 {} \;
find experiments/cache -path "*XXX*log.err" -exec tail -40 {} \;

# The exca-submitted pickle
find experiments/cache -name "*XXX_submitted.pkl"
```

## Common mistakes

- **Expecting `--submit` to block** — it returns immediately. Check status with `squeue`.
- **Not bumping version after code changes** — gets stale cached result. Use `--infra.mode=force`.
- **Clearing cache unnecessarily** — if you changed a UID field, exca already creates a new folder.
- **Putting non-deterministic fields in UID** — `wandb_run_id`, `device`, `num_workers` should go in `_exclude_from_cls_uid` so cache hits on irrelevant changes.

## Our `_exclude_from_cls_uid` conventions

```python
_exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
    "device", "batch_size", "num_workers",  # hardware/perf
    "wandb_project", "wandb_run_name", "wandb_run_id",  # logging
    "force_overwrite",  # ops
    "persistent_workers", "pin_memory", "matmul_precision",  # perf
)
```

Changing any of these reuses the same cache.
