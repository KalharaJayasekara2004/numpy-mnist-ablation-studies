# Reproducibility Guide

## Environment

Install dependencies with `python -m pip install -r requirements.txt` in the repository root. The code uses NumPy for all neural-network calculations, pandas for CSV loading, Matplotlib for plots, and SciPy for Task 8 image augmentation.

## Dataset

The scripts require exactly two dataset files alongside the scripts: `mnist_train.csv` and `mnist_test.csv`. Each CSV has 785 columns: an integer label from 0 to 9 followed by 784 grayscale pixels. Some scripts accept files with an optional header. Keep the files local and provide the original dataset source in the README/report. The normal full-data split is 50,000 training, 10,000 validation, and 10,000 test images.

## Execution

Use `python mnist_tasks11_15_extended.py --list` to see available modes. Run a mode using `--run NAME`; for example, `--run bottleneck`. The manual ablation implementation `mnist_tasks01_10_core.py` records the currently selected config and run name in its top-level definitions. `mnist_task04_lr_schedules.py` covers schedules, and `mnist_task08_regularization.py` has its own `run_choice` selector.

The scripts write timestamped run folders under `Results/`, including histories and plots. `Results/` is excluded from version control to avoid uploading hundreds of redundant assets. Copy only the final comparison figures, specific run histories if necessary, and summary CSVs into `figures/` and `results/`.

## Hyperparameter tuning caveat

In Task 12, the reported winner came from 20 sampled grid-subset and 20 random-search configurations using 10k training samples and 5 epochs. The full grid contains 243 possible configurations. Report this as a *reduced grid-subset comparison*, not exhaustive search, and only use the test set to assess the validation-selected winner.

## Evaluation provenance caveat

The supplied confusion matrix and classification metrics are from a run with 97.10% test accuracy. That differs from the 97.88% Task 6 optimizer result. Avoid presenting these diagnostics as evidence for the 97.88% model unless you regenerate evaluation artifacts from that exact checkpoint.

## Attribution

The original assignment PDF, full CSV dataset, and any copyrighted external materials should be uploaded only if redistribution is allowed. Never commit authentication tokens, API keys, or local machine credentials.
