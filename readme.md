# NuLearn4j

Implemented basic machine learning algorithms in Java 8 with functional programming api.

## Linear
This package implements linear regression and logistic regression algorithm. The linear regression is tested on boston housing
dataset and get MSE: 21.79281416675817.

Logistic Regression is tested on spambase dataset with 10 fold cross validation, and get a average accuracy about 92.4%.

## Tree
This package implements binary decision tree and regression tree. The decision tree tests spambase with 10-fold cross validation
and get about 89 ~ 90% accuracy. The regression tree is testing on boston housing dataset with a 2-level regression tree(MSE: 24.282).