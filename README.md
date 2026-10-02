# What Do PII Extraction Benchmarks Rank? — code and data

Code, run protocol, and result summaries for the paper *What Do PII Extraction Benchmarks
Rank? An Audit of Ten Corpora with GLiNER Extractors*. The repository lets you:

1. convert the ten public PII corpora to one span format,
2. run the eight model-free audit probes and the model review of their flags,
3. train and evaluate the GLiNER2 models of the paper (locally or on [Modal](https://modal.com)),
4. re-run every analysis and rebuild every table and figure of the paper.

## Layout

| Path | Contents |
|---|---|
| `config/` | corpus registry (`ner_datasets.py`, `pii_audit_datasets.py`: repos, type descriptions, splits) and training defaults |
| `data/`, `models/`, `training/`, `evaluation/`, `components/` | data conversion, GLiNER2 training (`training/train.py`), evaluation with the strict scorer (`evaluation/evaluate.py`, `evaluation/metrics.py`), and the rule detector used by probe U |
| `gliner2/` | vendored GLiNER2 1.2.6 with local changes (see `gliner2/NOTICE`) |
| `scripts/` | corpus conversion, audit, Modal runner, and one script per analysis (table below) |
| `out_pii_baselines/` | result summaries that the paper's tables are generated from, and the model-review sheets |
| `paper/` | LaTeX source; `paper/generated/` is written by `scripts/audit_paper_tables.py` |
| `PROTOCOL.md` | the run protocol and deviation log, written before each round of runs (in Vietnamese; Sections 9.4–9.19 cover this paper) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -r requirements.txt
export HF_TOKEN=...                                   # read access to the source corpora
```

Run every command from the repository root, so that `import gliner2` resolves to the vendored copy.

## 1. Corpora

The ten source corpora are public on the Hugging Face Hub (paper, Table 6). The converted
copies used in the paper are stored under a Hub namespace written as `anonymous-org` in
`config/ner_datasets.py` and `config/pii_audit_datasets.py`; replace it with your own
namespace, then convert and push:

```bash
python scripts/prepare_pii_datasets.py --datasets privy,pii_500k,kiji,gretel --hf-user <your-namespace> --push
python scripts/chunk_tab.py --push      # TAB training copy, cut into <=1,000-word pieces (test unchanged)
```

Conversion verifies every span (`text[start:end] == value`), caps the splits at
50k / 2.5k / 5k documents, and drops types with fewer than 20 training mentions.

## 2. Audit and model review

```bash
python scripts/pii_audit.py --table                 # probes S, L, C, U, G, O, B, IAA
python scripts/pii_audit_sample.py --extend 50      # draw the flag sample (already in out_pii_baselines/audit_validation)
python scripts/pii_audit_sample.py --passes out_pii_baselines/audit_validation/model_pass{1,2,3}_all.csv
```

`out_pii_baselines/audit_validation/` holds the sampled flags (`sample.csv`), the blind,
re-ordered item files given to each review pass (`items_*`), the three answer sheets
(`model_pass*`), and the stratified scores (`validation_scores.json`).

## 3. Training and evaluation

Locally, for one run:

```bash
python training/train.py --model fastino/gliner2-multi-v1 --ner-dataset privy --epochs 3 \
    --batch-size 8 --grad-accum 2 --seed 42 --output-dir ckpt/privy-s42 --no-push   # objective (a)
python evaluation/evaluate.py --model ckpt/privy-s42/best --ner-dataset privy --dataset <namespace>/pii-privy \
    --split test --use-desc --schema-case original --no-extra-val --threshold 0.5 --output-dir eval/privy-s42
```

The flags of objectives (b), (c), and (a′) are in `RECIPES` of `scripts/modal_pii_baselines.py`.
All runs of the paper were launched on Modal:

```bash
python scripts/launch_pii_jobs.py --profile <modal-profile> --deploy --recipe bce_bundle --families gliner2 \
    --datasets privy,pii_500k,kiji,gretel --seeds 42,1337
python scripts/launch_pii_jobs.py --profile <modal-profile> --status
```

Recipes: `bce` (a), `bce_bundle` (b), `set` (c), `bce_fullgrid` (a′), and `zeroshot`
(evaluation only; `--model` selects a released checkpoint, `--label-names` the type naming).
The runner stops a training run at the first skipped batch or NaN loss. GPU types and batch
sizes per corpus are in `GPU_OVERRIDE` / `BATCH_OVERRIDE`.

## 4. Analyses, tables, and figures

| Paper | Script | Output |
|---|---|---|
| Training objectives, Tables 8–16 | `scripts/rare_pii_tables.py`, `scripts/pii_sensitivity.py`, `scripts/pii_ablation_fullgrid.py` | `out_pii_baselines/rare_pii_tables.json`, `pii_sensitivity.json`, `pii_ablation_fullgrid.json` |
| AI4Privacy corpora (Sec. 5.1) | `scripts/pii_sensitivity_ai4p.py` | `pii_sensitivity_ai4p.json` |
| Released systems, Table 4 | `scripts/pii_offshelf_ranking.py` | `pii_offshelf_ranking.json` |
| Transfer test, Figure 1 | `scripts/pii_transfer_matrix.py`, `scripts/paper_fig_transfer.py` | `pii_transfer_matrix.json`, `paper/figs/paper/fig_transfer.pdf` |
| Why trained models lose, metric and margin checks | `scripts/pii_review_checks.py` | `pii_review_checks.json` |
| Renaming and naming studies | `scripts/pii_labelpara.py`, `scripts/pii_naming.py --report` | `pii_labelpara.json`, `pii_naming.json` |
| Untrained model | `scripts/pii_zeroshot_analysis.py` | `pii_zeroshot_analysis.json` |
| Compute budget | `scripts/pii_compute_budget.py` | `pii_compute_budget.json` |

These scripts read prediction dumps (`out_pii_baselines/<protocol>/<family>/<corpus>/seed-<s>/<split>/predictions/all.jsonl`),
which are released separately: <!-- prediction dumps: link to be added -->.

The summaries in `out_pii_baselines/` are enough to rebuild the paper without the dumps:

```bash
python scripts/audit_paper_tables.py     # -> paper/generated/*.tex
bash paper/build_pii_audit.sh            # -> paper/acl2023_pii_audit.pdf
```

## Third-party code

`gliner2/` is derived from [GLiNER2](https://github.com/fastino-ai/GLiNER2) 1.2.6, licensed under
the Apache License 2.0 (`gliner2/LICENSE`); our modifications are listed in `gliner2/NOTICE`.
Each source corpus keeps its own license; see the paper's appendix for sources.
