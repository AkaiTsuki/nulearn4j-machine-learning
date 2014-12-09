package net.nulearn4j.neighbor;

import net.nulearn4j.util.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class PolyKernel implements Kernel {
    private double degree = 3;
    private double C = 1;

    public PolyKernel() {
    }

    public PolyKernel(double degree, double c) {
        this.degree = degree;
        C = c;
    }

    @Override
    public double score(List<Double> x, List<Double> z) throws Exception {
        return -Math.pow(MathUtil.dot(x, z) + C, degree);
    }

    @Override
    public String toString() {
        return "PolyKernel{" +
                "degree=" + degree +
                ", C=" + C +
                '}';
    }
}
