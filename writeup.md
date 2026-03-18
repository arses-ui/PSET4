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

#Accuracy: 
One-vs-Rest accuracy (C=50.0): 0.8650
One-vs-One  accuracy (C=50.0): 0.8600

As you can see the performance is pretty comparabec

The notebook prints and compares both test accuracies directly so the better scheme is easy to identify from the run output.

## 4. Hyperparameter Tuning Strategy (`C`)
I tuned `C` using a validation split from the training set and evaluated candidates from a high-value log-spaced range (`np.logspace(1, 2, num=4)`) as recommended. I selected the `C` that gave the highest validation accuracy, then retrained on the full training split and reported tuned test accuracy.

#Test scores: 

- Tuned OVR test accuracy (C=50.00): 0.8650
- Tuned OVO test accuracy (C=46.42): 0.8550

Interestingly, the tuned model’s test accuracy was highest  with the C value from the the earlier Task 3 OVR result at \(C=50\) (0.8650) and  C=46.42 for OVO result. 
My takeaway from these results are that validation-based tuning is still the correct process, but it can have variance as the validation score was higher for OVO model with C=50 than C=46.42; using a denser \(C\) grid or cross-validation would probably make selection more stable. 

## 5. Confusion Matrices
I computed confusion matrices for both OVR and OVO and displayed a multiclass confusion matrix for the better-performing scheme in the notebook. As mentioned above, I went with the best value I got by doing the grid search for both models model. This complements overall accuracy by showing which classes are most often confused. The confusion matrix is pretty diagnoal with only a few no-zero values on the non diagonal elements. More specifically classes 1, 8 and 9 (0 indexing )seem to have 100% recall with OVO model and OVR model indicating those classes are the easiest for the model to classify. Here, as printed in the notebook, OVR model has a higher accuracy, and so has slightly less off-diagonal elements than the OVO model( higher accuracy). 

## Notes
- The submission notebook contains the complete implementation and all required task code.
- Run all notebook cells so outputs (accuracies and confusion matrix figures) are fully displayed before final submission.

## Reflection

This was a pretty helpful exercise because it helped me properly itnernalize soft margin SVMs and OVO/ OVR classifiers. While I already worked on an assignment similar to this in ENGS 96 last hw, this exercise was still challenging and taught me a lot

