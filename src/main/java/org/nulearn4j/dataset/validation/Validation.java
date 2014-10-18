package org.nulearn4j.dataset.validation;

import java.util.List;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class Validation {

    public static double mse(List<Double> predicts, List<Double> target) {
        double error = 0.0;
        for (int i = 0; i < predicts.size(); i++) {
            Double p = predicts.get(i);
            Double t = target.get(i);
            error += (p - t) * (p - t);
        }
        return error / predicts.size();
    }

    public static ConfusionMatrix confusionMatrix(List<Double> predicts, List<Double> target) {
        ConfusionMatrix cm = new ConfusionMatrix();
        for (int i = 0; i < target.size(); i++) {
            double p = predicts.get(i);
            double t = target.get(i);
            if (t == 1.0) {
                if (p == t) {
                    cm.tp += 1;
                } else {
                    cm.fn += 1;
                }
            } else {
                if (p == t) {
                    cm.tn += 1;
                } else {
                    cm.fp += 1;
                }
            }
        }
        return cm;
    }

    public static class ConfusionMatrix {
        public int tp;
        public int fp;
        public int tn;
        public int fn;

        public double accuracy() {
            int total = tp + fp + tn + fn;
            return 1.0 * (tn + tp) / total;
        }

        public double error() {
            return 1.0 - accuracy();
        }

        @Override
        public String toString() {
            return "ConfusionMatrix{" +
                    "tp=" + tp +
                    ", fp=" + fp +
                    ", tn=" + tn +
                    ", fn=" + fn +
                    ", error=" + error() +
                    ", accuracy=" + accuracy() +
                    '}';
        }
    }
}
