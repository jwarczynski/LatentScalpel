---
tags: [meta]
last_updated: 2026-05-03
---

# Schema — How this wiki works

This wiki is a persistent knowledge base for the GenieSAE project. It's maintained by LLM agents based on conversations, code inspection, experiment results, and external sources. Humans curate and ask questions; agents do the bookkeeping.

## Directory layout

```
wiki/
├── schema.md       This file — conventions and workflows
├── index.md        Catalog: all pages with one-line summaries
├── log.md          Append-only chronological log
│
├── models/         Model-specific pages (PLAID, GENIE, T5, fine-tuned variants)
├── pipeline/       SAE analysis pipeline stages (collection → training → interp)
├── concepts/       Reusable concepts (mup, rotary, VDM, Top-K SAE, etc.)
├── bugs/           Bugs found and how they were fixed
├── experiments/    Per-run pages (config, hyperparams, outcome, wandb link)
├── infrastructure/ Cluster, git, exca, wandb, disk
├── decisions/      Design choices with rationale
└── related/        Sibling projects (saescope, ShortcutFM)
```

## Page conventions

**Every page starts with YAML frontmatter:**

```yaml
---
tags: [plaid, training, bug]
status: fixed           # open | in-progress | fixed | wontfix (bugs/decisions only)
date: 2026-03-24        # creation or primary event date
related: [[model-X]]    # wiki-links to related pages
---
```

Frontmatter is Obsidian Dataview-friendly. Keep tags lowercase-hyphenated.

**Wiki links** use Obsidian style: `[[page-name]]` (file name without `.md`). Refer to files by stem, not path.

**Code references** use repo-relative paths so agents can read them directly:
`geniesae/plaid_model.py:_apply_mup_shapes`.

**Commit references** use short SHA: ``d06f5dd``.

**Wandb runs:** include full URL or run ID plus project.

## Page types

**Model page** — architecture, weights, checkpoints, known issues, evaluation results, related experiments.

**Pipeline page** — stage of the SAE analysis. Input/output, commands, configs, gotchas.

**Concept page** — reusable idea. Definition, why it matters here, references.

**Bug page** — symptom, root cause, fix (with commit SHA), how to detect, status. Keep these even after fixed — future agents need to know why the code looks the way it does.

**Experiment page** — one entry per training/eval run. Config file, hyperparameters, wandb link, outcome, what we learned.

**Decision page** — a non-obvious choice we made. What alternatives were considered, why we picked this one, what would change our minds.

**Infrastructure page** — external systems (cluster, git, wandb). Credentials referenced by name, not value.

## Workflows

### Ingesting a new source or conversation

1. Identify which pages are affected (read `index.md`, scan related areas).
2. Update those pages in place — don't create duplicates.
3. Append to `log.md` with `## [YYYY-MM-DD] ingest | <source>`.
4. Update `index.md` if new pages were created or existing summaries changed.

### Answering a question

1. Read `index.md` to find candidate pages.
2. Read candidates in full.
3. Synthesize answer. Cite pages via `[[page-name]]`.
4. If the answer is substantive, consider creating a new page under `experiments/`, `concepts/`, or a dedicated analysis location. **Good answers should be filed back into the wiki.**

### Lint pass (periodic)

Check for:
- Contradictions between pages (e.g. two pages disagreeing on a hyperparameter).
- Orphan pages (no inbound links).
- Stale claims (newer experiments overrode an older conclusion).
- Missing concept pages (something referenced everywhere but no dedicated page).
- Dead links (`[[broken]]`).

### Creating a new page

Use the existing naming pattern. Prefer descriptive kebab-case: `plaid-finetuned-v3b.md`, not `pf_v3b.md`. Keep one concept per page.

## Style

- Present tense for current state ("The model uses muP scaling").
- Past tense for experiments and decisions ("We tried lr=1e-5 but observed…").
- No marketing language. Be specific. Include numbers (loss values, step counts, dates).
- Prefer bullet points for enumerations, prose for reasoning.
- Include relevant commit SHAs, job IDs, wandb run IDs.
- If something is uncertain or speculative, mark it as such.

## What this wiki is not

- **Not a tutorial.** It's for agents and humans who already know the domain.
- **Not API docs.** Code docstrings serve that purpose.
- **Not a notebook.** Prefer stable pages over ad-hoc analysis dumps. Ad-hoc work goes in `log.md` with links to any artifacts.
