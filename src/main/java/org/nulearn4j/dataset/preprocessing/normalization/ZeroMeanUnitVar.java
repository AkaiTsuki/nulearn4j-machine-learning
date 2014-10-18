package org.nulearn4j.dataset.preprocessing.normalization;

import org.nulearn4j.dataset.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class ZeroMeanUnitVar implements Normalization<Double> {
    private List<Double> means;
    private List<Double> stds;

    public ZeroMeanUnitVar() {
        means = new ArrayList<>();
        stds = new ArrayList<>();
    }

    @Override
    public void setUpMeanAndVariance(Matrix<Double> matrix){
        int[] d = matrix.getDimension();
        int n = d[1];

        // calculate means and variances
        for (int i = 0; i < n; i++) {
            List<Double> column = matrix.getColumn(i);
            double mean = column.stream().mapToDouble(v -> v).average().getAsDouble();
            means.add(mean);
            stds.add(Math.sqrt(variance(column, mean)));
        }
    }

    @Override
    public void normalize(Matrix<Double> matrix) {
        int[] d = matrix.getDimension();
        int m = d[0];
        int n = d[1];

        for(int i=0; i< n; i++){
            for(int j =0; j<m; j++){
                matrix.set(j, i, (matrix.get(j,i) - means.get(i)) / stds.get(i));
            }
        }
    }

    private double variance(List<Double> vals, double mean) {
        double var = 0.0;
        for (Double val : vals) {
            var += (val - mean) * (val - mean);
        }
        return var / vals.size();
    }
}
