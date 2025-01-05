# Explaining ML Model Predictions

The goal of this project was to understand why a model produced a given prediction. Models are never completely accurate and understanding incorrect predictions at a deep level can lead to new avenues for model refinement or data preparation. As a use case in this project a Random Forest Classifier with default parameters was trained using the Breast Cancer Wisconsin (diagnostic) dataset from Scikit-learn.

The out-of-the-box RF classifier performs very well on the data, as seen in the confusion matrix:

![confusion_matrix](/visualizations/confusion_matrix.jpg)

However, it could be argued that predicting `Benign` when in reality the observation is `Malignant` is an important error to minimize. Therefore, understanding what led to the predictions for those observations would be worthwhile.

Looking at feature importances is a straightforward way to understand what components the model used most to determine predictions overall:

![feature_importance](/visualizations/feature_importance.jpg)

This doesn't provide the resolution needed to understand what led to the incorrect predictions. To address this, I used the underlying model object to reconstruct the trees in the forest and compare the paths of False Negatives (FN), True Positives (TP), and True Negatives (TN) using the following procedure:

1. Create pairwise combinations of:
    * FNs and TPs
    * TPs to themselves
    * TNs and TPs
2. For each pairwise combo:
    1. Trace paths through trees and find where nodes are different
    2. Back track one step to find divergence points
    3. Calculate difference between observations of divergence feature
3. Aggregate differences to identify features where FNs were more similar to TNs than TPs at divergence points using the following formula:
    * abs(FNvTP diff - TPvTP diff) - abs(FNvTP diff - TNvTP diff)
  
The following chart shows the degree to which features may contribute to incorrect predictions:

![feature_difference](/visualizations/feature_difference.jpg)

The features with the greatest difference would be candidates for further analysis. However, each feature is not guaranteed to be considered in each tree. For this reason I continued the analysis to compare "Good" trees (contributed to incorrect predictions in 1 or 2 out of 5 total errors) and "Bad" trees (contributed to incorrect predictions in all 5 errors). For this analysis I compared the proportion features were considered in "Good" trees to "Bad" trees:

![feature_consideration](/visualizations/feature_consideration.jpg)

In the above chart, the further away a feature is from the diagonal, the more heavily it was considered in "Good" trees or "Bad" trees. Features considered more often in "Bad" trees would be candidates for further analysis. 

Importantly, `mean_concavity` was a top feature in Feature Importance, Feature Difference, and Proportion Consideration. Since 3 different methods all point to `mean_concavity`, this would be strong evidence that it should be studied in greater detail for how it could contribute to incorrect predictions.

## Conclusions

Analyzing incorrect predictions at a deep level using this analysis exposes possible sources of error in a way distinct from other aggregate methods of model evaluation. Furthermore, finding concensus between traditional and new methods presented here could greatly aid in model improvement in subsequent iterations.

---

To install `graphviz`:
1. `brew install graphviz`
2. `pip install graphviz`
