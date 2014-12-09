package net.nulearn4j.validation;

import net.nulearn4j.util.Statistic.MathUtil;

import java.util.*;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class Validation {

    public static class Point implements Comparable<Point> {
        double x;
        double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public int compareTo(Point o) {
            int cmp = Double.compare(this.x, o.x);
            if (cmp != 0) return cmp;
            return Double.compare(this.y, o.y);
        }
    }

    public static double auc(List<Point> points) {
        Collections.sort(points);
        double auc = 0.0;
        for(int i=1; i<points.size(); i++){
            Point p1 = points.get(i-1);
            Point p2 = points.get(i);
            double h = p2.x - p1.x;
            auc += (p1.y + p2.y) * h * 0.5;
        }
        return auc;
    }

    public static List<Point> roc(List<Double> actualLabels, List<Double> score, double positiveLabel, double negativeLabel) {
        List<Point> roc = new LinkedList<>();

        double positive = 0.0;
        double negative = 0.0;
        for (Double l : actualLabels) {
            if (l == positiveLabel) {
                positive += 1;
            } else {
                negative += 1;
            }
        }

        List<Integer> args = MathUtil.argsort(score, true);
        List<Double> sortedActualLabels = new ArrayList<Double>(actualLabels.size());
        for (Integer i : args) {
            sortedActualLabels.add(actualLabels.get(i));
        }
        double tp = 0.0, fp = 0.0;
        roc.add(new Point(fp, tp));

        for (Double l : sortedActualLabels) {
            if (l.equals(positiveLabel)) {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            roc.add(new Point(fp / negative, tp / positive));
        }
        return roc;
    }

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
