package org.nulearn4j.tree;

import org.apache.log4j.BasicConfigurator;
import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.validation.Validation;
import org.nulearn4j.util.Statistic.DoubleListStatistic;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by jiachiliu on 10/19/14.
 */
public class RegressionTree extends Cart {

    public RegressionTree(int[] features, int label) {
        super(features, label);
    }

    public RegressionTree(int[] features, int label, int maxLevel, int minDataInTreeNode) {
        super(features, label, maxLevel, minDataInTreeNode);
    }

    @Override
    protected double calculateScore(Matrix<Double> matrix) {
        List<Double> target = matrix.getColumn(label);
        double mean = DoubleListStatistic.mean(target);
        double sse = 0.0;
        for (Double t : target) {
            sse += (t - mean) * (t - mean);
        }
        return sse;
    }

    @Override
    protected double calculateScore(Matrix<Double> left, Matrix<Double> right, double parentScore) {
        return parentScore - (calculateScore(left) + calculateScore(right));
    }

    @Override
    protected double majorityVote(Matrix<Double> matrix) {
        List<Double> target = matrix.getColumn(label);
        return DoubleListStatistic.mean(target);
    }

    @Override
    protected boolean isSameLabel(Matrix<Double> matrix) {
        return false;
    }

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        Matrix<Double> train = DatasetLoader.loadBostonHousingTrain("\\s+");
        int label = train.getColumnCount() - 1;

        Matrix<Double> test = DatasetLoader.loadBostonHousingTest("\\s+");
        List<Double> testTarget = test.getColumn(label);
        test = test.removeColumn(label);

        int[] features = IntStream.range(0, train.getColumnCount() - 1).toArray();
        Cart classifier = new RegressionTree(features, label, 2, 5);
        classifier.fit(train);
        List<Double> predicts = classifier.predict(test);
        double mse = Validation.mse(predicts, testTarget);
        System.out.println("Test MSE: " + mse);
    }


}
