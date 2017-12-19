The paper, An Analysis of Single-Layer Networks in Unsupervised Feature Learning , introduces a novel approach to learn features on images using much simpler systems than the current state-of-art deep neural networks. The paper was authored by Adam Coates, Andrew Ng, Honglak Lee and published in 2011 at the International Conference on Artificial Intelligence and Statistics conference. According to the results reported in the paper, the methods developed by the researchers are able to achieve the same level of accuracy as deep neural networks.

In this project, I implemented the Gaussian Mixture model from scratch in Python to learn features of images in an unsupervised settings. The dataset is downloaded from Kaggle (The link to download data: https://www.kaggle.com/c/state-farm-distracted-driver-detection). 

Step 1 - Initialization: Instead of randomly picking data points as the initial centroids, we run one iteration of k-means to get the initial cluster centroids and use them as the means for the number of clusters determined. We also initialize the covariance matrices of each cluster to be identity matrices.

Step 2 - Expectation: evaluate the probability of each data point to see how likely it belongs to the different cluster. We use the probability density function of a multivariate Gaussian with the initialized means and covariance matrices.

Step 3 - Maximization: update parameters, which include means, covariance matrices, and prior probabilities.

Step 4 - Check for convergence of the log-likelihood if the epsilon is smaller than 0.01. 

