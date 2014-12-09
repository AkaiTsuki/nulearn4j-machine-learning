package net.nulearn4j.neighbor;

import net.nulearn4j.util.Statistic.MathUtil;

import java.util.List;

/**
 * Created by jiachiliu on 12/5/14.
 */
public class DotProductKernel implements Kernel {
    @Override
    public double score(List<Double> x, List<Double> z) throws Exception {
        return MathUtil.dot(x, z);
    }
}
