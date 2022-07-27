## AIM : Identify fraudulent credit card transactions
### Dataset Used : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Description : The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## About Anomaly Detection:
Anomaly detection is a common problem that comes up in many applications such as credit card fraud detection, network intrusion detection, identifying malignancies in the heath care domain and so on.
It is the process of finding the outliers or anomalous points given a dataset. In these applications, usually there are many examples of normal data points, but very few or no examples of anomalous data points. In other words we have mostly examples of a single class and have very few examples of the anomaly class making the classification problem highly imbalanced.
Hence supervised learning techniques such as random forests and SVM are hard to use in this highly imbalanced setting.

We use unsupervised learning techniques which deal well with imbalanced datasets. The result of each algo will be a vactor indicating if the data point passed for prediction is an anomaly or not.

### 1.) Isolation Forests:
The random forest classifier is an ensemble learning technique. It consists of a collection of decision trees, whose outcome is aggregated to come up with a prediction.
Individual decision trees are prone to overfitting. Random Forest is a bagging technique that constructs multiple decision trees by selecting a random subset of data points and features for each tree. Given a subset of the data and featurees, the decision trees themselves are created by partitioning the data feature by feature until each leaf is homogeneous.

The goal of isolation forests is to “isolate” outliers. The algorithm is built on the premise that anomalous points are easier to isolate tham regular points through random partitioning of data.
The algorithm itself comprises of building a collection of isolation trees(itree) from random subsets of data, and aggregating the anomaly score from each tree to come up with a final anomaly score for a point.
The isolation forest algorithm is explained in detail in the video above. Here is a brief summary.

Given a dataset, the process of building or training an isolation tree involves the following:

- Select a random subset of the data
- Until every point in the dataset is isolated:
- selecting one feature at a time
- Partition the feature at a random point in its range.
- An interior point requires more partitions to isolate, while an outlier point can be isolated in just a few partitions.

Given a new point, the prediction process involves:

- For Each itree in the forest
- Perform binary search for the new point across the itree, traversing till a leaf
- Compute an anomaly score based on the depth of the path to the leaf
- Aggregate the anomaly score obtained from the individual itrees to come up with an overall anomaly score for the point.
- Anamolous points will lead to short paths to leaves, making them easier to isolate,  while interior points on an average will have a significantly longer path to the leaf.

LINKS:
https://machinelearninginterview.com/topics/machine-learning/explain-isolation-forests-for-anomaly-detection/
https://medium.com/grabngoinfo/isolation-forest-for-anomaly-detection-cd7871ae99b4



### 2.) One Class SVM:
One-Class Support Vector Machine (SVM) is an unsupervised model for anomaly or outlier detection. Unlike the regular supervised SVM, the one-class SVM does not have target labels for the model training process. Instead, it learns the boundary for the normal data points and identifies the data outside the border to be anomalies.

### 3.) Local Outlier Factor(LOF) Algorithm
The LOF algorithm is an unsupervised outlier detection method which computes the local density deviation of a given data point with respect to its neighbors. It considers as outlier samples that have a substantially lower density than their neighbors.
The number of neighbors considered, (parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers. In practice, such informations are generally not available, and taking n_neighbors=20 appears to work well in general.

Local Outlier Factor (LOF) is an algorithm for finding points that are outliers relative to their k nearest neighbors. Informally, the algorithm works by comparing the local density of a point to the local densities of its k nearest neighbors. Points with local densities lower than their neighbors will be classified as outliers.
The primary hyperparameter under analyst control in LOF is k, the number of neighbors. The scikit-learn documentation offers some guidance for selecting k, but notes that the information needed to make an informed choice is seldom available in advance and that k=20 is usually a good choice. That said, one useful guideline is that the minimum value of k should be at least the size of the smallest known cluster in the data, such that local outliers can be detected around that cluster (otherwise parts of the cluster itself could be classified as outliers).

How does it work?
The main idea behind LOF is that points with local densities lower than their neighbors can be considered outliers.
Suppose k=20. Take some point A, with point A’s 20 nearest neighbors clustered closely around A. Then A is close to its neighbors, and A’s neighbors are close to their neighbors, so A’s local density is close to the local densities of its neighbors. A is not likely to be classified as an outlier.
Suppose, on the other hand, that A’s 20 nearest neighbors are clustered closely together, but that they are relatively far from point A. In other words, A’s neighbors are all close to each other, but A is not close to any of its neighbors. Then A’s neighbors will all have similar (high) local densities, but point A will have a low local density. Consequently, A will likely be classified as an outlier.

increasing k tends to decrease the number of points labeled as outliers.

LINKS: https://innerjoin.bit.io/local-outlier-factor-analysis-with-scikit-learn-fc89372ee658



## Steps followed in the notebook:

1.) EDA

2.) Data Cleaning

3.) Model building

4.) Model Evaluation (accuracy not a good measure as we have imbalanced datasets. Recall indicates % of anomalies that were correctly captured).
We can see IsolationForests performs the best.
