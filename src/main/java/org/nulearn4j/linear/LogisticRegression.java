package org.nulearn4j.linear;

import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;
import org.nulearn4j.dataset.preprocessing.normalization.Normalization;
import org.nulearn4j.dataset.preprocessing.normalization.ZeroMeanUnitVar;
import org.nulearn4j.dataset.validation.Validation;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class LogisticRegression extends LinearRegression {

    public LogisticRegression(){
        super();
    }

    public LogisticRegression(double learning_rate, double converged, double maxIteration){
        super(learning_rate, converged, maxIteration);
    }

    @Override
    protected double delta(double predict, double actual, double x) {
        return learning_rate * (predict - actual) * predict * (1.0 - predict) * x;
    }

    @Override
    public Double predict(Row<Double> row) {
        return sigmoid(super.predict(row));
    }

    private Double sigmoid(double val) {
        return 1.0 / (1 + Math.exp(-val));
    }

    public List<Double> predictToBinary(Matrix<Double> test, double threshold) {
        List<Double> predicts = super.predict(test);

        return predicts.stream().map(v -> {
            if (v <= threshold) return 0.0;
            else return 1.0;
        }).collect(Collectors.toList());
    }

    public static void main(String[] args) throws Exception{
        Matrix<Double> spambase = DatasetLoader.loadSpambase(",");
        // Shuffle the data since it put all spam at the head fo file.
        spambase.shuffle();

        Matrix<Double>[] splits = spambase.split(spambase.getRowCount() / 10);
        Matrix<Double> test = splits[0];
        Matrix<Double> train = splits[1];

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

        LogisticRegression classifer = new LogisticRegression(0.001, 0.0001, 100);
        classifer.fit(train, trainTarget);
        List<Double> predictLabels = classifer.predictToBinary(test, 0.5);
        System.out.println(Validation.confusionMatrix(predictLabels,testTarget));
    }
}
