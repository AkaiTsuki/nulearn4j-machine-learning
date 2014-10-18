# NuLearn4j

Implemented basic machine learning algorithms in Java 8 with functional programming api.

## Linear
This package implements linear regression and logistic regression algorithm. The linear regression is tested on boston housing
dataset and get MSE: 21.79281416675817.

Logistic Regression is tested on spambase dataset. The dataset is shuffled first and splitted into train set and test set.
The test result is about the following:

ConfusionMatrix{tp=169, fp=15, tn=252, fn=24, error=0.08478260869565213, accuracy=0.9152173913043479}