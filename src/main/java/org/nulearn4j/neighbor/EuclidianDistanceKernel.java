package org.nulearn4j.neighbor;

import org.nulearn4j.util.Statistic.MathUtil;

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
