Random Forest is an ensemble of many decision trees. Each tree overfits in a different way (thanks to randomness), and averaging their predictions cancels out a lot of that overfitting—giving strong, stable performance with minimal tuning.

## Why it’s a good fit for house prices
- Captures nonlinearities and feature interactions (e.g., lot area × neighborhood).
- Robust to outliers and monotone transformations; little to no scaling needed.
- Handles many features, including sparse one-hot encoded categoricals.

Limitations: Doesn’t extrapolate beyond the range seen in training; large forests can be memory-heavy; raw impurity-based importances can be biased toward high-cardinality features (prefer permutation importance for reliability).
