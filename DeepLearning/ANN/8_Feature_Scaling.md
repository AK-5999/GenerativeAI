

# Feature Scaling Study Notes

## 1. Introduction to Feature Scaling

Feature scaling is a data preprocessing step used to transform the numerical features of a dataset into a similar, standardized scale. When features have vastly different ranges (e.g., Age: 20–70 vs. Salary: $15,000–$200,000), machine learning models can struggle to perform optimally.

### Why is Feature Scaling Necessary?

* **Gradient Descent Convergence:** In algorithms like Linear Regression, Logistic Regression, and Neural Networks, features with larger magnitudes cause the gradient descent step to oscillate, drastically slowing down convergence. Scaling makes the error surface more spherical, allowing faster convergence.
* **Distance-Based Algorithms:** Algorithms like K-Nearest Neighbors (KNN), K-Means Clustering, and Support Vector Machines (SVM) calculate the Euclidean distance between points. Unscaled features with larger ranges will disproportionately dominate the distance calculations.
* **Feature Importance / Weights:** In models using regularization (L1/L2), unscaled features will unfairly penalize weights, as coefficients are highly sensitive to the scale of the input data.

> ⚠️ **Note:** Tree-based algorithms (Decision Trees, Random Forests, Gradient Boosting, XGBoost) are **invariant** to feature scaling because they split nodes based on threshold rules, not distances or gradients.

---

## 2. Standardization (Z-Score Normalization)

Standardization centers the data around a mean of 0 with a standard deviation of 1. It transforms the distribution so that it takes the shape of a standard normal distribution.

### Formula

For each feature value $x$, the standardized value $x'$ is calculated as:


$$x' = \frac{x - \mu}{\sigma}$$


Where:

* $\mu$ = Mean of the feature column
* $\sigma$ = Standard deviation of the feature column

### Key Characteristics

* **Outliers:** Standardization does not bound the data to a specific range (like 0 to 1). If there are extreme outliers, they will remain outliers but will be brought closer to the center, meaning standardization is relatively robust to them.
* **When to use:** Use when the data follows a Normal/Gaussian distribution, or when using algorithms that assume normally distributed data (like Linear Discriminant Analysis).

---

## 3. Normalization Techniques

Normalization rescales the features into a fixed range—most commonly $[0, 1]$.

### A. Min-Max Scaling

This is the most common form of normalization. It shifts and rescales the data so that all values fall strictly between 0 and 1.

* **Formula:** 
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$


* **Impact of Outliers:** Highly sensitive to outliers. A single extreme outlier will compress all the normal data points into a very small, squashed range.

### B. Mean Normalization

Instead of subtracting the minimum value, this method centers the data by subtracting the mean. The resulting values fall between $-1$ and $1$.

* **Formula:** 
$$x' = \frac{x - \mu}{x_{max} - x_{min}}$$



### C. MaxAbs Scaling

This scales the data by dividing each value by the maximum absolute value in the feature. It forces the data to fall between $-1$ and $1$.

* **Formula:** 
$$x' = \frac{x}{|x_{max}|}$$


* **When to use:** It is highly useful for **sparse data** (datasets with mostly zeros) because it preserves the zeros in the dataset.

### D. Robust Scaling

If your dataset contains a high amount of extreme outliers that you cannot remove, Robust Scaling is the ideal choice. It scales the data using the median and the Interquartile Range (IQR) rather than the mean and standard deviation/min-max values.

* **Formula:** 
$$x' = \frac{x - \text{median}}{IQR} = \frac{x - Q_2}{Q_3 - Q_1}$$



Where:
* $Q_1$ = 25th percentile
* $Q_2$ = Median (50th percentile)
* $Q_3$ = 75th percentile

---

## 4. Normalization vs. Standardization: How to Choose?

| Metric / Scenario | Standardization | Normalization (Min-Max) |
| --- | --- | --- |
| **Output Range** | No fixed boundary (typically $[-3, 3]$ for normal data) | Strictly bounded (usually $[0, 1]$) |
| **Handling Outliers** | Fairly robust; preserves outlier properties safely | Highly sensitive; compresses good data if outliers exist |
| **When to use** | Most Machine Learning algorithms (Linear Reg, Logistic Reg, KNN, SVM, PCA) | Deep Learning / Neural Networks (where inputs must be bounded), or Image processing (pixels $0\text{--}255$) |
| **Distribution Assumption** | Works best if data is normally distributed | Does not assume any specific data distribution |

### Pro-Tip Checklist

1. **Rule of Thumb:** Most of the time, try **Standardization** first as it behaves predictably with most machine learning algorithms.
2. If you know the exact minimum and maximum bounds of your features (e.g., image pixels $0$ to $255$), use **Min-Max Scaling**.
3. If your data has unmanageable outliers, use **Robust Scaling**.

---

## 🔗 Original Resources

* [Video 1: Feature Scaling - Standardization (Day 24)](https://youtu.be/1Yw9sC0PNwY?si=450935odGnh0sZLi) | [Associated GitHub Code](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day24-standardization)
* [Video 2: Feature Scaling - Normalization (Day 25)](https://youtu.be/eBrGyuA2MIg?si=k5RTdjteBzHblns-) | [Associated GitHub Code](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day25-normalization)

