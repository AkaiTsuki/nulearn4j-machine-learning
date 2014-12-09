package net.nulearn4j.neighbor;

import net.nulearn4j.util.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class EuclidianDistanceKernel implements Kernel {
    @Override
    public double score(List<Double> x, List<Double> z) throws Exception {
        return MathUtil.distance(x, z);
    }

    public String toString() {
        return "EuclidianDistanceKernel";
    }
}
