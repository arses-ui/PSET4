# Assignment 4 Write-Up: Support Vector Machines

## 1. Binary SVM with a Nonlinear Kernel
I implemented a soft-margin SVM in dual form and solved it with `cvxopt.solvers.qp`. The quadratic program uses:
- objective: minimize `0.5 * a^T (TKT) a - 1^T a`
- constraints: `0 <= a_i <= C` and `sum_i a_i t_i = 0`

For the nonlinear kernel, I implemented `rbf`, `poly`, and `sigmoid` options and used a polynomial kernel in the multiclass experiments.

## 2. Predictive Model
After solving for Lagrange multipliers, I selected support vectors and computed the intercept `b` from margin support vectors using the KKT-based relation. Predictions are made from the sign of the decision function:
`f(x) = sum_i a_i t_i K(x_i, x) + b`.

## 3. One-vs-Rest vs One-vs-One
I implemented both voting schemes:
- One-vs-Rest (OVR): train one binary classifier per class.
- One-vs-One (OVO): train one classifier per class pair and use majority vote (with confidence tie-break).

The notebook prints and compares both test accuracies directly so the better scheme is easy to identify from the run output.

## 4. Hyperparameter Tuning Strategy (`C`)
I tuned `C` using a validation split from the training set and evaluated candidates from a high-value log-spaced range (`np.logspace(1, 2, num=4)`) as recommended. I selected the `C` that gave the highest validation accuracy, then retrained on the full training split and reported tuned test accuracy.

## 5. Confusion Matrices
I computed confusion matrices for both OVR and OVO and displayed a multiclass confusion matrix for the better-performing scheme in the notebook. This complements overall accuracy by showing which classes are most often confused.

## Notes
- The submission notebook contains the complete implementation and all required task code.
- Run all notebook cells so outputs (accuracies and confusion matrix figures) are fully displayed before final submission.
