# Experiment Index

This index maps the written assignment's Tasks 1–14 to code and evidence. The repository stores an initial subset of your provided plots and results. **Add the remaining files from your Windows `Results/` directory** before submission.

| Task | Topic | Relevant script / CLI mode | Evidence to publish |
| --- | --- | --- | --- |
| 1 | Activation functions | `mnist_tasks01_10_core.py` or `--run activations` | Validation curves, activation derivatives, accuracy table |
| 2 | Network depth | `mnist_tasks01_10_core.py` or `--run depth` | Depth accuracy and gradient curves |
| 3 | Network width | `mnist_tasks01_10_core.py` or `--run width` | Accuracy, parameter counts, time-vs-width |
| 4 | Learning rates and schedules | `mnist_tasks01_10_core.py`, `mnist_task04_lr_schedules.py` | LR comparison, schedule comparison |
| 5 | Batch size | `mnist_tasks01_10_core.py` or `--run batch_size` | Batch-size accuracy and time comparison |
| 6 | Optimizers | `mnist_tasks01_10_core.py` or `--run optimizers` | 6 optimizer × 2 LR comparison and gradient curves |
| 7 | Weight initialization | `mnist_tasks01_10_core.py` or `--run initialization` | Weight/activation histograms, validation curves |
| 8 | Regularization | `mnist_task08_regularization.py` | 8-run comparison: baseline, L2, dropout, augmentation, early stopping |
| 9 | Loss functions | `mnist_tasks01_10_core.py` or `--run loss_functions` | Three output/loss combinations, convergence table |
| 10 | Data preprocessing | `mnist_tasks01_10_core.py` or `--run preprocessing` | Four-method comparison and statistics |
| 11 | Architecture variations | `--run residual`, `bottleneck`, `multi_branch`, `output_activation` | Four pairwise comparisons with time/parameter counts |
| 12 | Hyperparameter tuning | `--run tuning` | `results/task12_tuning/`, validation-accuracy plot |
| 13 | Debugging and sanity checks | `--run debugging` | `results/task13_debugging/`, first-layer weights |
| 14 | Evaluation and error analysis | `--run error_analysis` or main evaluation | `results/task14_error_analysis/`, confusion matrix, ROC, errors |

`--run` arguments refer to `python mnist_tasks11_15_extended.py --run NAME`.

## Highlights from the reported experiments

| Task | Best observed test accuracy / principal result | Note |
| --- | --- | --- |
| 1 | ReLU: 94.96% | Base shallow architecture |
| 2 | 5 hidden layers: 97.32% | Tested up to 10 hidden layers |
| 3 | Width 1024: 97.00% | Greater compute cost |
| 4 | Constant LR 0.1: 97.58% | 20-epoch comparison |
| 5 | Batch size 16: 97.17% | More updates per epoch |
| 6 | RMSProp, LR 0.001: 97.88% | Highest observed in these reported runs |
| 7 | He initialization: 94.75% | Two hidden layers, 10 epochs |
| 8 | Data augmentation: 97.59% | Baseline 97.45% |
| 9 | CE + Softmax: 94.96% | Compared with MSE alternatives |
| 10 | Per-pixel standardization: 96.22% | Mean raw pixel = 33.3168, std = 78.5644 |
| 11 | Wide reference network: 97.47% | Bottleneck: 97.31% with fewer parameters |
| 12 | Random-search winner: 95.52% test | 5 epochs, 10k training examples |
| 13 | Gradient-check error: 3.43e-10 | 10-example memorization reached 100% |
| 14 | Evaluation artifact accuracy: 97.10% | Source checkpoint should be identified if known |

**Do not compare these results as if they share one fixed training budget or architecture.** Each result is meaningful within its own task's controlled comparison.
