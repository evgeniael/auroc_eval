# A Tale of Two Error Categories: Exploring Concealed Trade-Offs in the Errors of Automated Judges in Evaluation of Uncertainty Quantifiers

Investigating how automated correctness judgements impact scores of uncertainty quantifiers in terms of AUROC and other selective prediction metrics. The pipeline generates samples, scores correctness under multiple criteria, computes uncertainty scores, and stores everything under `results/` for analysis.

This code supplements an accepted paper at INLG 2026. Link to the paper to be provided shortly.

Supplementary code for the Bayesian Analysis described in the paper can be found [here](https://github.com/probabll/buqeval).

## Repository layout

```
auroc_eval/
├── scripts/
│   ├── pipeline.py      # Main entrypoint: generation → correctness → uncertainty
│   └── annotate.py      # Interactive human annotation of greedy predictions
├── utils/
│   ├── datasets.py
│   ├── generations.py
│   ├── correctness_criteria.py
│   └── quantifiers.py
├── results/
│   ├── generations/
│   ├── correctness/
│   ├── uncertainty_quantifiers/
│   ├── annotations/
│   ├── plots/
│   ├── rank_correlations/
│   └── meta-analysis/
└── analysis_notebooks/  # Downstream analysis of stored results
```

## Requirements

- A GPU is recommended (models are loaded with Hugging Face Transformers).
- Hugging Face access for gated models (e.g. Llama).
- For GPT-based judges, set `OPENAI_API_KEY` (e.g. in a `.env` file loaded by `python-dotenv`).

## Running the pipeline

`scripts/pipeline.py` is the main script. It runs three stages in order and writes checkpoints under `results/`:

1. **Generation** — greedy and unbiased samples for each question  
2. **Correctness** — automated correctness metrics labels for greedy answers  
3. **Uncertainty** — various uncertainty quantifiers (e.g. entropy, semantic entropy, P(True))  

Stages resume from existing JSON files when they are already present and complete.

```bash
python scripts/pipeline.py --dataset trivia_qa --model llama-8b-instruct --size_test_set 100 --seed 0
```

### Useful arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | `trivia_qa`, `ambig_qa_single_answer`, or `ambig_qa_multiple_qas` | `trivia_qa` |
| `--model` | Generation model (`llama-8b-instruct`, `llama-3b-instruct`, `qwen-7b-instruct`, `qwen-0.5b-instruct`) | `llama-8b-instruct` |
| `--llm_as_judge` | Models that can be used as correctness judges | `llama-8b-instruct`, `gpt-5.4-mini`, `qwen-0.5b-instruct` |
| `--nli_model` | NLI model for semantic entropy | `deberta-large` |
| `--num_unbiased_samples` | Number of unbiased samples per question | `10` |
| `--size_test_set` | Number of evaluation questions | `100` |
| `--seed` | Random seed | `0` |
| `--max_new_tokens` | Max decode length per generation call | `300` |
| `--checkpoint_every` | Write JSON every N items (`1` = every item) | `1` |

Threshold ranges for token F1, BLEU, ROUGE-L, embedding similarity, and BERTScore can also be passed via the corresponding `--*_threshold_range` flags.

## Human annotation

Use `scripts/annotate.py` to label greedy predictions interactively. Progress is saved after each item and sessions resume automatically.

```bash
python scripts/annotate.py --input results/generations/<file>.json
```

Optional flags:

- `--output` — annotation JSON path (default: under `results/annotations/`, derived from the input filename)
- `--start_index` — 0-based index to start from (overrides automatic resume)

Labels: `y` (correct), `n` (incorrect), `u` (unsure), `s` (skip).

## Results

Pipeline outputs are named:

```text
{dataset}_{model}_{size_test_set}_{seed}.json
```

and stored under:

- `results/generations/`
- `results/correctness/`
- `results/uncertainty_quantifiers/`

Human annotations from `annotate.py` are stored in `results/annotations/`. Analysis artifacts (plots, meta-analysis tables etc.) live in the corresponding subfolders under `results/`.

## Analysis notebooks

Notebooks in `analysis_notebooks/` analyse existing results:

| Notebook | Purpose |
|---|---|
| `analyse_auroc.ipynb` | AUROC of uncertainty quantifiers across correctness criteria |
| `ablation_bootstrapping.ipynb` | Bootstrapping ablations |
| `ablation_sp_metrics.ipynb` | Other selective prediction metrics ablations |
| `analyse_metastudy.ipynb` | Meta-analysis of related studies |

Open them from the repo root (or adjust paths in the notebooks) after the relevant result files exist.
