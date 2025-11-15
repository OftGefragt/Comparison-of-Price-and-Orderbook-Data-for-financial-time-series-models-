# GA-Optimized Extended Isolation Forest (EIF) with SVM Post-Processor for Anomaly Detection
This repository implements an anomaly detection pipeline that combines Extended Isolation Forest (EIF) with a post-processing One-Class SVM, with the EIF hyperparameters optimized via a Genetic Algorithm (GA).

## Overview

Unlike standard Isolation Forest, Extended Isolation Forest (EIF) uses randomly oriented hyperplanes instead of axis-aligned splits. This produces more uniform anomaly scores and reduces artifacts caused by feature correlations or data rotation.

The original study reports similar computational cost but more robust results compared to Isolation Forest.

**Reference Paper:** Extended Isolation Forest – IEEE

![Comparison of Isolation Forest Variants](imgs/blobs.png)
*Img by: HARIRI ET AL.: EXTENDED ISOLATION FOREST*

## Extension Level Parameter

The algorithm-specific parameter **extension level** refers to a number in the range `[0, P − 1]`, where `P` is the number of features.

- **0** corresponds to the standard Isolation Forest
- **P − 1** represents full extension, accounting for all features

This parameter helps reduce bias and prevents the algorithm from favoring a particular axis, making the model rotation-invariant and more robust.

![Sample Branch Hyperplanes](imgs/hyperplanes.png)
*Img by: HARIRI ET AL.: EXTENDED ISOLATION FOREST*
*C is identical to the regular Isolation Forest.*

## Genetic Algorithm Optimization

A Genetic Algorithm (GA) is a metaheuristic inspired by the process of natural selection. It is used here to discover the best combination of parameters for EIF, such as:

- Number of trees
- Sample size
- Contamination rate
- Extension level


## License

Personal contributions marked under MIT license, please refer to eif/license.txt for license for extended isolation forest.

## References

* https://ieeexplore.ieee.org/document/8888179
* https://github.com/sahandha/eif
* https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html

## Using the Ready Files

You can directly load and use the pretrained EIF and One-Class SVM models:

```python
import pickle

with open("models/eif_model.pkl", "rb") as f:
    forest = pickle.load(f)

with open("models/ocsvm_augmented.pkl", "rb") as f:
    oc_svm = pickle.load(f)

eif_scores = forest.compute_paths(X_test)
svm_preds = oc_svm.predict(X_test)









