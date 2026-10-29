# Weekly Challenge 40: Unsupervised Anomaly Detection (Isolation Forest from Scratch)

## Description
For Week 40, I implemented an **Isolation Forest** from scratch using pure Python and NumPy. 

When working with massive educational databases to predict dropout risks, algorithms often try to model what a "normal" student looks like in order to spot the abnormal ones. Isolation Forest flips this logic entirely: it completely ignores the normal profiles and aggressively hunts for the anomalies by attempting to isolate them.

## How it works (The Math)
Because anomalies are "few and different," they are much easier to isolate using random spatial cuts.
1. **Random Splitting:** The algorithm builds multiple trees by selecting a random feature and a random threshold between the minimum and maximum values.
2. **Path Length ($h(x)$):** As the tree splits the data, anomalies are isolated almost immediately, ending up in very short branches near the root of the tree. Normal students, being clustered together, require dozens of random cuts to be separated, ending up at the bottom of the tree.
3. **Anomaly Score:** The algorithm calculates the average path length across all trees $E(h(x))$ and normalizes it using the mathematical constant for average tree depth $c(n)$.
   $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
   * If the score is close to $1$, the profile is a definitive anomaly.
   * If the score is much smaller than $0.5$, the profile is normal.

## Technical Highlights
* **High Efficiency:** Unlike DBSCAN or K-Means, Isolation Forest does not require computationally expensive Euclidean distance calculations, making it $O(n)$ in time complexity and perfect for massive datasets.
* **Pure Math Implementation:** Built the binary search tree logic and the $c(n)$ Euler-Mascheroni depth adjustment formula directly from the original 2008 research paper without relying on `scikit-learn`.

## Dependencies
* Python 3.14.3
* NumPy
