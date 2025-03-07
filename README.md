# Bayesian-OvR-Multiclass

**Machine learning homework by a Sophomore in SDUFE**

Binary classifier Bayesian in multiclassification mission,(contrast with GussianNB)

**Dataset:UCI-iris(150 samples)**
***
#### **Explanation:**

High-dimensional Gaussian Distribution Probability Formula:

$$p(x) = \frac{1}{(2\pi)^{\frac{d}{2}}|\sum|^\frac{1}{2}}e^{[-\frac{1}{2}(x-\mu)^T]\sum^{-1}(x-\mu)}$$

Likelihood Calculation:

$$p(x|w) = \frac{1}{(2\pi)^{\frac{d}{2}}|\sum_\omega|^{\frac{1}{2}}}e^{([-\frac{1}{2}(x-\mu_\omega)^T\sum^{-1}_{\omega}(x-\mu_\omega)])}$$

(Where $\sum_\omega$ represents the covariance matrix, $|\sum_\omega|$ represents the determinant of the covariance matrix, and $\sum^{-1}_{\omega}$ is the inverse of the covariance matrix).

***
**Calculation Process Declaration:**

1. Calculate the difference function between the data point and the mean vector:

$$ d = x - \mu_\omega$$

2. Calculate the normalization constant:

$$ a = \frac{1}{(2\pi)^{\frac{d}{2}}|\sum_\omega|^\frac{1}{2}}$$

3. Calculate the parameter of the exponential function (para: quadratic form):

$$ e_{arg} = ([-\frac{1}{2}(x-\mu_\omega)^T\sum^{-1}_ {\omega}(x-\mu_\omega)])$$

4. Calculate the exponential function value:

$$e_{formula} = e^{e_{arg}}$$

5. Combine the formulas:

$$result = ae_{formula}$$

**Principal Component Analysis: Mahalanobis Distance**

It is an effective method for calculating the similarity between two sets of unknown samples. Unlike Euclidean distance, it takes into account the correlations between various features.

Mahalanobis Distance Definition:

$$D_M(\overrightarrow{x}) = \sqrt{(\overrightarrow{x} - \overrightarrow{\mu})^T\textstyle{\sum^{-1}}(\overrightarrow{x}-\overrightarrow{\mu})}$$

If the covariance matrix is an identity matrix, the Mahalanobis distance simplifies to the Euclidean distance.

***

**Result Output Processing Strategies: (OvR Strategy and OvO Strategy)**

**1. OvR Strategy: Directly select the classifier with the highest positive probability (requires 2N classifiers).**

**2. OvO Strategy: Compare different classifications and select the final label (requires N(N-1)/2 classifiers).**
