package net.nulearn4j.linear;

import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.core.matrix.Row;
import net.nulearn4j.util.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 12/5/14.
 */
public class Perceptron {

    List<Double> weights;

    public double predictOne(List<Double> x) throws Exception {
        return MathUtil.dot(weights, x);
    }

    private void initWeights(int n) {
        weights = MathUtil.zeros(n);
    }

    public void fit(Matrix<Double> train, List<Double> target) throws Exception {
        flip(train, target);
        int m = train.getRowCount();
        int n = train.getColumnCount();
        initWeights(n);
        int l = 0;
        int c;
        do {
            c = 0;
            l++;
            for (int i = 0; i < m; i++) {
                Row<Double> r = train.getRow(i);
                double p = predictOne(r.getData());
                if (p <= 0) {
                    c += 1;
                    MathUtil.add(weights, r.getData());
                }
            }
            System.out.format("iteration %d: total mistakes %d, weights: %s\n", l, c, weights);
        } while (c > 0);

    }

    private boolean allPossitive(Matrix<Double> train) throws Exception {
        for (Row<Double> r : train.getRows()) {
            if (predictOne(r.getData()) <= 0) return false;
        }
        return true;
    }

    private void flip(Matrix<Double> train, List<Double> target) {
        for (int i = 0; i < target.size(); i++) {
            if (target.get(i) < 0) {
                Row<Double> r = train.getRow(i);
                for (int j = 0; j < r.size(); j++) {
                    r.set(j, -r.get(j));
                }
                target.set(i, -target.get(i));
            }
        }
    }
}
