package org.nulearn4j.boost;

import org.nulearn4j.boost.learner.OptimalLearner;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;
import org.nulearn4j.validation.Validation;
import org.nulearn4j.util.Statistic.MathUtil;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jiachiliu on 11/11/14.
 */
public class AdaBoost {

    private int maxRound;
    private List<RoundResult> weakLearners;

    public AdaBoost() {
        maxRound = 100;
        weakLearners = new LinkedList<>();
    }

    public AdaBoost(int round) {
        maxRound = round;
        weakLearners = new LinkedList<>();
    }

    public void fit(Matrix<Double> train, List<Double> trainTarget, Matrix<Double> test, List<Double> testTarget) {
        int m = train.getRowCount();
        int n = train.getColumnCount();
        List<Double> weights = initWeights(m);
        int round = 0;

        OptimalLearner learner = new OptimalLearner();
        List<Double> trainPredicts = MathUtil.zeros(trainTarget.size());
        List<Double> testPredicts = MathUtil.zeros(testTarget.size());

        while (round < this.maxRound) {
            learner.fit(train, trainTarget, weights);
            int f = learner.getFeature();
            double t = learner.getThreshold();
            double weightedError = learner.getWeightedError();
            double confidence = 0.5 * Math.log((1.0 - weightedError) / weightedError);
            this.weakLearners.add(new RoundResult(f, t, weightedError));

            // Predict on train data
            List<Double> trainPredict = this.hypothesis(train, f, t);
            List<Double> trainPredictVal = MathUtil.multiply(trainPredict, confidence);
            MathUtil.add(trainPredicts, trainPredictVal);
            List<Double> trainPredictLabels = sign(trainPredicts);
            Validation.ConfusionMatrix trainCM = validate(trainPredictLabels, trainTarget);

            // Predict on test data
            List<Double> testPredict = this.hypothesis(test, f, t);
            List<Double> testPredictVal = MathUtil.multiply(testPredict, confidence);
            MathUtil.add(testPredicts, testPredictVal);
            List<Double> testPredictLabels = sign(testPredicts);
            Validation.ConfusionMatrix testCM = validate(testPredictLabels, testTarget);
            double auc = Validation.auc(Validation.roc(testTarget, testPredicts, 1.0, -1.0));

            System.out.format("Round %2d, feature %2d, threshold %f, round error: %f, train error: %f, test error: %f, auc: %f\n",
                    round, f, t, weightedError, trainCM.error(), testCM.error(), auc);
            this.updateWeights(weights, trainPredict, trainTarget, weightedError);

            round++;
        }
    }

    private Validation.ConfusionMatrix validate(List<Double> predictLabels, List<Double> actualLabels) {
        return Validation.confusionMatrix(predictLabels, actualLabels);
    }

    public List<Double> sign(List<Double> vals) {
        List<Double> s = new ArrayList<>(vals.size());

        for (Double d : vals) {
            if (d <= 0) s.add(-1.0);
            else s.add(1.0);
        }
        return s;
    }

    public List<RoundResult> getWeakLearners() {
        return weakLearners;
    }

    private List<Double> initWeights(int m) {
        List<Double> weights = new ArrayList<>(m);

        for (int i = 0; i < m; i++)
            weights.add(1.0 / m);

        return weights;
    }

    private void updateWeights(List<Double> weights, List<Double> predicts, List<Double> targets, double weightedErr) {
        double z = 2.0 * Math.sqrt(weightedErr * (1.0 - weightedErr));
        double frac = weightedErr / (1.0 - weightedErr);
        for (int i = 0; i < weights.size(); i++) {
            double tmp = Math.pow(frac, predicts.get(i) * targets.get(i));
            tmp = Math.sqrt(tmp);
            double w = weights.get(i);
            double updated = (w * tmp) / z;
            weights.set(i, updated);
        }
    }

    private List<Double> hypothesis(Matrix<Double> data, int f, double t) {
        List<Double> predicts = new ArrayList<>(data.getRowCount());
        for (Row<Double> r : data.getRows()) {
            if (r.get(f) <= t)
                predicts.add(-1.0);
            else
                predicts.add(1.0);
        }
        return predicts;
    }

    public static class RoundResult {
        public int feature;
        public double threshold;
        public double confidence;

        public RoundResult(int feature, double threshold, double confidence) {
            this.feature = feature;
            this.threshold = threshold;
            this.confidence = confidence;
        }
    }
}
