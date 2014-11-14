package org.nulearn4j.linear;

import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;
import org.nulearn4j.dataset.preprocessing.normalization.Normalization;
import org.nulearn4j.dataset.preprocessing.normalization.ZeroMeanUnitVar;
import org.nulearn4j.validation.Validation;
import org.nulearn4j.validation.Validation.ConfusionMatrix;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class LogisticRegression extends LinearRegression {

    public LogisticRegression() {
        super();
    }

    public LogisticRegression(double learning_rate, double converged, double maxIteration) {
        super(learning_rate, converged, maxIteration);
    }

    public List<Double> predictToBinary(Matrix<Double> test, double threshold) {
        List<Double> predicts = super.predict(test);

        return predicts.stream().map(v -> {
            if (v <= threshold) return 0.0;
            else return 1.0;
        }).collect(Collectors.toList());
    }

    @Override
    public Double predict(Row<Double> row) {
        return sigmoid(super.predict(row));
    }

    @Override
    protected double delta(double predict, double actual, double x) {
        return learning_rate * (predict - actual) * predict * (1.0 - predict) * x;
    }

    private Double sigmoid(double val) {
        return 1.0 / (1 + Math.exp(-val));
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
            ConfusionMatrix cm = run(test, train);
            accuracy += cm.accuracy();
            error += cm.error();
        }

        System.out.println("Average Accuracy: " + accuracy / k + " Average error: " + error / k);
    }

    public static void randomSplit() throws Exception {
        Matrix<Double> spambase = DatasetLoader.loadSpambase(",");
        // Shuffle the data since it put all spam at the head fo file.
        spambase.shuffle();

        Matrix<Double>[] splits = spambase.split(spambase.getRowCount() / 10);
        Matrix<Double> test = splits[0];
        Matrix<Double> train = splits[1];

        run(test, train);
    }

    private static ConfusionMatrix run(Matrix<Double> test, Matrix<Double> train) {
        int label = train.getColumnCount() - 1;
        List<Double> trainTarget = train.getColumn(label);
        List<Double> testTarget = test.getColumn(label);
        train = train.removeColumn(label);
        test = test.removeColumn(label);

        Normalization<Double> norm = new ZeroMeanUnitVar();
        norm.setUpMeanAndStd(train);

        norm.normalize(train);
        train.addColumn(0, 1.0);
        norm.normalize(test);
        test.addColumn(0, 1.0);

        LogisticRegression classifier = new LogisticRegression(0.001, 0.0001, 100);
        classifier.fit(train, trainTarget);
        List<Double> predictLabels = classifier.predictToBinary(test, 0.5);
        ConfusionMatrix cm = Validation.confusionMatrix(predictLabels, testTarget);
        System.out.println(cm);
        return cm;
    }
}
