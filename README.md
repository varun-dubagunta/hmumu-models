# hmumu-models

Training pipeline for H->mumu signal/background classifiers with k-fold cross-validation. Supports DNN, Transformer (focal loss), and Transformer (cross-entropy) architectures.

## Setup

```bash
pip install tensorflow pandas scikit-learn numpy
```

## Files

- `config.py` -- features, hyperparams, paths, process definitions
- `model.py` -- transformer architecture, GatherLayer, focal loss
- `model_dnn.py` -- feedforward DNN architecture
- `train.py` -- single-split training loop (used internally by kfold)
- `kfold.py` -- k-fold CV with event-ID grouping
- `evaluate.py` -- plotting and evaluation from prediction files

Data files (.pkl) and trained weights (.keras) are not tracked. Place your data in `data/` before running.

## K-fold training

```bash
python src/kfold.py --model dnn_focal \
    --dataset v2602c_samples_2024 \
    --data data/v2602c_samples_2024.pkl \
    --k 5
```

Model choices: `dnn`, `dnn_focal`, `transformer_focal`, `transformer_ce`

Splits on `FullEventId` so no event leaks between folds. For each fold i, fold i is held-out test, fold (i+1)%k is validation, the rest is training. After all folds, predictions are combined so every event is scored exactly once by a model that never trained on it.

Output goes to `data/<dataset>/<model>/`:
- `predictions_with_metadata.pkl` -- combined predictions with fold column
- `predictions_df.pkl` -- full softmax probability matrix
- `fold_assignments.pkl` -- event-to-fold mapping
- `fold0/`, `fold1/`, ... -- per-fold predictions

## Evaluate

```bash
python src/evaluate.py --model dnn_focal --dataset v2602c_samples_2024
```

Reads from the combined predictions files produced by kfold.py.

## Config

Edit `src/config.py` to change input features (`DATA_COLUMNS`), transformer token groupings (`TOKEN_GROUPS`), signal/background process definitions, hyperparameters, and per-class sample weight targets.