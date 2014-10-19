package org.nulearn4j.tree;

import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.preprocessing.normalization.Normalization;
import org.nulearn4j.dataset.preprocessing.normalization.ZeroMeanUnitVar;
import org.nulearn4j.dataset.validation.Validation;
import org.nulearn4j.util.Statistic.DoubleListStatistic;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class DecisionTree extends Cart {

    public DecisionTree(int[] features, int label) {
        super(features, label);
    }

    public DecisionTree(int[] features, int label, int maxLevel, int minDataInTreeNode) {
        super(features, label, maxLevel, minDataInTreeNode);
    }

    @Override
    protected double calculateScore(Matrix<Double> matrix) {
        double info = 0.0;
        double total = 1.0 * matrix.getRowCount();
        List<Double> target = matrix.getColumn(label);
        for (Double l : getUniqueLabels(matrix)) {
            long count = target.stream().filter(v -> v.equals(l)).count();
            info += -(count / total) * DoubleListStatistic.log2((count / total));
        }
        return info;
    }

    @Override
    protected double calculateScore(Matrix<Double> left, Matrix<Double> right, double parentScore) {
        double total = 1.0 * (left.getRowCount() + right.getRowCount());
        double scoreLeft = (left.getRowCount() / total) * calculateScore(left);
        double scoreRight = (right.getRowCount() / total) * calculateScore(right);
        return parentScore - scoreLeft - scoreRight;
    }

    @Override
    protected double majorityVote(Matrix<Double> matrix) {
        long max = 0;
        List<Double> target = matrix.getColumn(label);
        double majority = target.get(0);

        for (Double l : getUniqueLabels(matrix)) {
            long count = target.stream().filter(v -> v.equals(l)).count();
            if (max < count) {
                max = count;
                majority = l;
            }
        }

        return majority;
    }

    @Override
    protected boolean isSameLabel(Matrix<Double> matrix) {
        List<Double> target = matrix.getColumn(label);
        Double value = target.get(0);
        for (int i = 1; i < target.size(); i++) {
            if (!value.equals(target.get(i))) return false;
        }
        System.out.println("data splits has same label: " +target.get(0));
        return true;
    }

    private Set<Double> getUniqueLabels(Matrix<Double> matrix) {
        Set<Double> labels = new HashSet<>();
        matrix.getRows().stream().forEach(r -> labels.add(r.get(label)));
        return labels;
    }

    public static void main(String[] args) throws Exception {
        kfold();
    }

    public static void kfold() throws Exception {
        Matrix<Double> spambase = DatasetLoader.loadSpambase(",");
        // Shuffle the data since it put all spam at the head fo file.
        spambase.shuffle();

        int k = 10;

        double accuracy = 0.0;
        double error = 0.0;

        for (int i = 0; i < k; i++) {
            System.out.println("\n============== Fold " + i + "=================");
            Matrix<Double> train = spambase.kFoldTrain(k, i);
            Matrix<Double> test = spambase.kFoldTest(k, i);
            Validation.ConfusionMatrix cm = run(test, train);
            accuracy += cm.accuracy();
            error += cm.error();
        }

        System.out.println("Average Accuracy: " + accuracy / k + " Average error: " + error / k);
    }

    private static Validation.ConfusionMatrix run(Matrix<Double> test, Matrix<Double> train) {
        int label = train.getColumnCount() - 1;
        List<Double> testTarget = test.getColumn(label);

        int[] features = IntStream.range(0,train.getColumnCount()-1).toArray();
        System.out.println(Arrays.toString(features));
        Cart classifier = new DecisionTree(features, label);
        classifier.fit(train);
        Cart.printTree(classifier.getRoot());
        List<Double> predictLabels = classifier.predict(test);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predictLabels, testTarget);
        System.out.println(cm);
        return cm;
    }
}
