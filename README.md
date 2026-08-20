# Group23_Assignment1_code — CS601T Deep Learning, Assignment 1

Perceptron (logistic / tanh / linear activation), trained with batch
gradient descent **implemented entirely from scratch** — no
scikit-learn, no Keras/PyTorch/TensorFlow, no library perceptron or
gradient-descent routines. `numpy` is used only as an array container
and for basic linear algebra (dot products), `matplotlib` only for
plotting, `openpyxl` only for reading the regression `.xlsx` files.

## 1. Put your data in place

```
data/Group23/classification/LS_Group23/Class1.txt   (x1 x2 per line)
data/Group23/classification/LS_Group23/Class2.txt
data/Group23/classification/LS_Group23/Class3.txt
data/Group23/classification/NLS_Group23.txt          (see note below)
data/Group23/regression/Univariate/Univariate.xlsx    (columns: x, y)
data/Group23/regression/Bivariate/Bivariate.xlsx      (columns: x1, x2, y)
```

**NLS file format**: `preprocessing.load_nls_data()` expects the first
line to be the per-class example counts (e.g. `500 500` for 2 classes,
or `500 500 500` for 3), followed by the data rows in that class
order — matching "the number of examples in each class and their
order is given at the beginning of each file" from the assignment
sheet. If your actual file's header line looks different, adjust
`load_nls_data()` in `src/preprocessing.py` (it also has a fallback
path for a plain `x1 x2 label` format — see the docstring).

Also check `main.py`'s call:
```python
X_nls, y_nls = pp.load_nls_data(NLS_PATH, class_labels=(1, 2))
```
change `class_labels=(1, 2)` to `(1, 2, 3)` if your NLS dataset has 3
classes.

## 2. Verify the data loads correctly

Before a full run, sanity-check shapes/labels:

```bash
python main.py --check-data
```

## 3. Run everything

```bash
pip install numpy matplotlib openpyxl
python main.py
```

This trains, for every dataset, **both** required activations
(logistic and tanh for classification; linear for regression), and
writes every plot + metric required by the assignment under
`results/`:

```
results/classification/LS/{logistic,tanh}/
    error_vs_epoch.png                  # per pairwise classifier
    decision_region_C<i>_vs_C<j>.png    # one per class pair
    decision_region_combined.png
    metrics.json                        # confusion matrix, accuracy,
                                         # precision/recall/F per class + means
results/classification/NLS/{logistic,tanh}/   (same structure)

results/regression/Univariate/
    error_vs_epoch.png
    fit_train.png, fit_test.png         # model output vs target
    scatter_train.png, scatter_test.png # target (x) vs model output (y)
    metrics.json                        # RMSE / %RMSE, train + test
results/regression/Bivariate/                  (same structure, 3D fit plots)

results/comparison/overall_summary.json
```

## 4. Design notes / where the assignment's requirements are met

- **From scratch**: `src/perceptron.py` implements the net input,
  activation, error, and the batch delta-rule gradient-descent weight
  update by hand (see the module docstring for the exact update rule).
  No ML/optimization library is imported anywhere in `src/`.
- **One-against-one multiclass**: `src/one_vs_one.py` trains one
  binary perceptron per class pair and combines them by majority vote
  (`OneVsOnePerceptron`), used for both the 3-class LS data and the
  2/3-class NLS data.
- **70/30 split**: `utils.train_test_split_per_class` splits *within
  each class* (not just overall) to match the assignment's per-class
  70/30 instruction; regression uses a plain 70/30 split
  (`train_test_split_simple`), since there are no classes there.
- **Metrics**: `src/metrics.py` computes the confusion matrix,
  accuracy, and per-class + mean precision/recall/F-measure by hand
  from the confusion matrix (no `sklearn.metrics`).
- **Report**: the `results/**/metrics.json` files and all PNGs are
  meant to be pulled directly into `Group23_Assignment1_report.pdf` —
  this repo only produces the code + raw results, not the PDF report
  itself (write the inferences/discussion by hand, per the assignment
  instructions).

## 5. Hyperparameters

Learning rate, epoch count, and random seed are set at the top of
`main.py` (`LR`, `EPOCHS`, `SEED`). The default `EPOCHS = 500` with
`LR = 0.01` converges well on well-scaled 2-D toy-style data; if your
real data has features on very different scales, standardize first
with `utils.standardize()` (already imported/available) before
training, and mention that preprocessing step in the report.

## 6. Tested with synthetic placeholder data

This pipeline was run end-to-end on synthetic stand-in data (3
well-separated Gaussian blobs for LS, a two-moons-style set for NLS,
and linear-plus-noise data for both regression tasks) to confirm every
script, plot, and metric works before you drop in the real Group 23
files. Swap in the real data and re-run — no code changes should be
needed unless your NLS header format differs (see section 1).
