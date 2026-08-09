## Experimental Results

The K-Nearest Neighbors (KNN) model was tested using three different values of k: 3, 5, and 7.

The experimental results are shown below:

| k Value | Accuracy |
|---|---:|
| 3 | 0.9130 or 91.30% |
| 5 | 1.0000 or 100.00% |
| 7 | 0.9565 or 95.65% |

Based on the experimental results, **k = 5** achieved the highest accuracy of **1.0000 or 100.00%**. Therefore, **k = 5** was selected as the best value for this dataset.

## Experimental Conclusion

This experiment applied the k-Nearest Neighbors (KNN) algorithm to classify animals using the Zoo Animals Extended Dataset. Three different values of k were tested: k = 3, k = 5, and k = 7.

The results showed that:

- k = 3 achieved an accuracy of 91.30%.
- k = 5 achieved an accuracy of 100.00%.
- k = 7 achieved an accuracy of 95.65%.

Among the tested values, **k = 5** produced the best performance with an accuracy of **100.00%**. This means that the model correctly classified all samples in the testing dataset.

The experiment demonstrates that the choice of k has a significant effect on the performance of the KNN model. A small k value may be more sensitive to individual data points or noise, while a larger k value may produce more generalized predictions.

In this experiment, **k = 5** was the most suitable value for classifying animals in the selected dataset. However, the results may vary if a different train-test split, random state, or dataset is used.
