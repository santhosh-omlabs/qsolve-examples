# Quantum-kernel classifier

Binary classification with a quantum-kernel SVM, tuned by cross-validation and compared against an
equally tuned classical RBF-SVM, logistic regression, a random forest and gradient boosting on a
held-out test set. The workflow is plain Qiskit (`src/workflow.py`); the quantum-kernel code is in
`src/kernel/`.

## Input
A built-in `dataset` (real: `breast_cancer`; synthetic: moons, circles, xor, blobs), or your own
`data` table (`X`, `y`), or explicit `train` and `test` tables. Up to 8 features (one qubit each)
and 60 training points.

### Real data: breast-cancer diagnosis
`breast_cancer` samples tumours (stratified, by `seed`) from the UCI Wisconsin Diagnostic Breast
Cancer dataset (569 tumours, 30 measurements, malignant or benign; CC BY 4.0, bundled with
scikit-learn) and keeps `features` principal components. The scaler and PCA are fitted on the
training rows only.

## How the quantum method works
1. Features are scaled to [0, pi] (training statistics only) and multiplied by a bandwidth.
2. Each point `x` becomes a state `|phi(x)>` with a ZZ feature map (`reps` layers of H, phase and
   RZZ gates).
3. The kernel `k(a, b) = |<phi(a)|phi(b)>|^2` is measured as the probability of all zeros after
   `U(a)` then `U(b)^dagger`. Every entry is one circuit; all circuits of a Gram matrix go through
   `ctx.sampler` in one call, so on a QPU they travel in as few batches as the batch limit allows.
4. A support vector machine with a precomputed kernel (C chosen by cross-validation) learns from
   the (positive semi-definite projected) training matrix and predicts the test points.
5. `reps` and `bandwidth` (0 = auto) are chosen by cross-validation on noise-free simulated kernels,
   which costs no QPU time.

Every method returns its predictions for the test points; the SDK re-scores them against the true
labels (the objective is the test error rate), so no method can claim a better number than its
predictions earn.

## What the mitigation report contains
Kernel entries are probabilities of one outcome, so mitigation is applied to the measured
distribution of each circuit:

- readout correction: the device's per-bit readout errors (when it reports them) are inverted on
  the full distribution; applied to every kernel entry;
- depolarizing rescale (QPU and `fake:` targets only): 8 mirror circuits (`U(a)` then `U(a)^dagger`,
  ideal P(all zeros) = 1) fit the strength of a global depolarizing channel, and every entry is
  rescaled by `(k - L/2^n) / (1 - L)` and clipped to [0, 1]. This is a model of the noise, not a
  guarantee; it is reported as such.

`ctx.report_mitigation` declares the output for **one representative circuit**: the training-pair
circuit whose exact kernel value is the median. Its reported distribution is the one after both
steps, `kept` is 1.0 (a feature map has no conserved quantity to post-select on), and the steps list
the fidelity to the circuit's ideal output after raw counts, readout correction and the rescale. The
SDK measures the raw fidelity of the same circuit itself. The whole-matrix quality is in the answer's
`kernel_check`: mean and max absolute error of the measured kernel against the exact one, with and
without mitigation, next to the error shot noise alone would give.

`kernel_mode: exact` computes the kernel by statevector simulation and uses it for the SVM. Because
the quantum method must really use the backend, it also runs 8 kernel circuits spread over the
kernel's value range on the target as a spot check, and reports mitigation for the median one.

## Output
Test accuracy with a 95% Wilson interval, the best classical accuracy, a paired exact McNemar test
against the tuned RBF-SVM, the confusion matrix, predictions of both models and the kernel check.

## Limits
- Small data only: kernel circuits grow with the square of the training set (60 points = 1770
  training circuits plus one per test point and training point).
- Quantum kernels do not generally beat tuned classical kernels; the comparison is reported
  honestly. One test point is several percentage points on small sets, so read small differences as
  "on par".
- Depolarizing rescale assumes global depolarizing noise; correlated or non-uniform noise is not
  corrected. Mitigation numbers are `reported`, the raw fidelity is measured.

## Run and test
```bash
cd qsolve-sdk
uv run qsolve run ../examples/quantum-kernel-classifier -i ../examples/quantum-kernel-classifier/examples/moons.json
uv run qsolve test ../examples/quantum-kernel-classifier
uv run pytest ../examples/quantum-kernel-classifier/tests
```
