# PCL Detection

Binary classification of Patronising and Condescending Language (PCL) in paragraphs about vulnerable communities. Based on the SemEval 2022 Task 4 dataset.

## Repository Structure

```
PCL-detection/
├── data/
├── experiments/
├── notebooks/
├── runs/
├── src/
└── requirements.txt
```

### data/

- `raw/` — Original dataset files from SemEval (dontpatronizeme_pcl.tsv, etc.)
- `processed/` — Cleaned train/dev CSVs ready for modelling
- `splits/` — Official SemEval paragraph ID splits
- `cache/` — Tokenized tensors cached to disk (speeds up training)

### experiments/

Jupyter notebooks for model development:

- `01_baseline.ipynb` — Baseline transformer model (paragraph-level classification)
- `02_albert_deberta_search.ipynb` — Hyperparameter search across ALBERT/DeBERTa
- `03_localised_pcl_token.ipynb` — Token-level classifier with span aggregation (main approach)
- `04_localised_pcl_window.ipynb` — Window-based pooling experiments
- `05_localised_pcl_sampling.ipynb` — Span sampling strategies

Also contains `outputs/` and `runs/` subdirectories with checkpoints from training runs.

**experiments/token-level/** — Final model outputs:
- `best_model.pt` — Trained model checkpoint
- `dev.txt` — Predictions on dev set (one label per line)
- `test.txt` — Predictions on test set (submission file)

### notebooks/

Exploratory data analysis:

- `EDA_basic_stat_profiling.ipynb` — Dataset size, class distribution, text lengths
- `EDA_lexical_analysis.ipynb` — Word frequencies, vocabulary analysis
- `EDA_noise_artifacts.ipynb` — Data quality issues, artifact detection
- `EDA_semantic_syntactic.ipynb` — Linguistic patterns in PCL text
- `EDA_with_PCL_categories.ipynb` — Analysis by PCL subcategory

### runs/

Saved model checkpoints and Optuna study results from hyperparameter tuning:

- `optuna_task1/` — Baseline model trials
- `optuna_stage04_localised_tokencls/` — Token classifier trials
- `optuna_stage05_span_paragraph_agg/` — Span aggregation trials
- `optuna_stage05_windowpool/` — Window pooling trials

### src/

Reusable Python modules:

**src/data/**
- `make_dataset.py` — Data loading, preprocessing, span extraction, submission file generation
- `span_sampler.py` — Negative span sampling strategies

**src/training/**
- `loss.py` — Custom loss functions (focal loss, class-weighted BCE)
- `metrics.py` — F1 computation and evaluation utilities
- `search_utils.py` — Optuna objective functions and search helpers
- `span_classifier.py` — Span-level classifier model and aggregation logic
- `tokenization_utils.py` — Tokenizer wrappers and BIO label alignment

## Setup

```bash
pip install -r requirements.txt
```

Use `requirements_mac.txt` for Apple Silicon or `requirements_baseline.txt` for minimal dependencies 
    - though this may only support you for EDA