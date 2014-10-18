package org.nulearn4j.dataset.preprocessing.normalization;

import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.util.Statistic.DoubleListStatistic;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jiachiliu on 10/18/14.
 *
 * Implements the zero mean unit variance normalization
 */
public class ZeroMeanUnitVar implements Normalization<Double> {
    private List<Double> means;
    private List<Double> stds;

    public ZeroMeanUnitVar() {
        means = new ArrayList<>();
        stds = new ArrayList<>();
    }

    @Override
    public void setUpMeanAndStd(Matrix<Double> matrix) {
        int[] d = matrix.getDimension();
        int n = d[1];

        // calculate means and variances
        for (int i = 0; i < n; i++) {
            List<Double> column = matrix.getColumn(i);
            double mean = DoubleListStatistic.mean(column);
            means.add(mean);
            stds.add(DoubleListStatistic.std(column, mean));
        }
    }

    @Override
    public void normalize(Matrix<Double> matrix) {
        int[] d = matrix.getDimension();
        int m = d[0];
        int n = d[1];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                matrix.set(j, i, (matrix.get(j, i) - means.get(i)) / stds.get(i));
            }
        }
    }
}
