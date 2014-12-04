package org.nulearn4j.neighbor;

import org.nulearn4j.util.Statistic.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class GaussianKernel implements Kernel {

    private double gamma = 5;

    public GaussianKernel(double gamma) {
        this.gamma = gamma;
    }

    public GaussianKernel() {
    }

    @Override
    public double score(List<Double> x, List<Double> z) throws Exception {
        List<Double> diff = MathUtil.minus(x, z);
        double distance = MathUtil.dot(diff, diff);
        return -Math.exp(-(distance / (2.0 * gamma * gamma)));
    }

    @Override
    public String toString() {
        return "GaussianKernel{" +
                "gamma=" + gamma +
                '}';
    }
}
