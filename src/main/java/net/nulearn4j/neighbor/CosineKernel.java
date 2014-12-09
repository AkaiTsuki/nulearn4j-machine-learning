package net.nulearn4j.neighbor;

import net.nulearn4j.util.Statistic.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class CosineKernel implements Kernel {
    @Override
    public double score(List<Double> x, List<Double> z) throws Exception {
        return -MathUtil.dot(x, z) / Math.sqrt(MathUtil.dot(x, x) * MathUtil.dot(z, z));
    }

    public String toString() {
        return "CosineKernel";
    }
}
