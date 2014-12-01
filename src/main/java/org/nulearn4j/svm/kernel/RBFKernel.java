package org.nulearn4j.svm.kernel;

import org.nulearn4j.util.Statistic.MathUtil;

import java.util.HashMap;
import java.util.List;

/**
 * Created by jiachiliu on 12/1/14.
 */
public class RBFKernel extends LinearKernel {

    private double gamma = 0.1;

    public RBFKernel(double gamma) {
        this.gamma = gamma;
    }

    public RBFKernel() {
        super();
    }

    @Override
    protected double dotProduct(List<Double> x1, List<Double> x2) throws Exception {
        List<Double> diff = MathUtil.minus(x1, x2);
        double distance = MathUtil.dot(diff, diff);
        return Math.exp(-(distance / (2.0 * gamma * gamma)));
    }
}
