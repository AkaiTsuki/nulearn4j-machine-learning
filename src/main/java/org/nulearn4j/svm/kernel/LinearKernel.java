package org.nulearn4j.svm.kernel;

import org.nulearn4j.util.Statistic.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 11/26/14.
 */
public class LinearKernel implements Kernel {
    private Double[][] cache;
    private int totalAccess;
    private int miss;

    public LinearKernel(int m) {
        cache = new Double[m][m];
    }

    public double transform(List<Double> x1, List<Double> x2, int i1, int i2) throws Exception {
        if (cache[i1][i2] == null) {
            cache[i1][i2] = MathUtil.dot(x1, x2);
            miss++;
        }
        totalAccess += 1;
        return cache[i1][i2];
    }

    public double missRate() {
        return 1.0 * miss / totalAccess;
    }
}
