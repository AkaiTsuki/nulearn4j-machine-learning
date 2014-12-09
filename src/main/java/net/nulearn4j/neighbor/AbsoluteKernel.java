package net.nulearn4j.neighbor;

import java.util.List;

/**
 * Created by jiachiliu on 12/6/14.
 */
public class AbsoluteKernel implements Kernel {

    Kernel kernel;

    public AbsoluteKernel(Kernel k) {
        kernel = k;
    }

    @Override
    public double score(List<Double> x, List<Double> z) throws Exception {
        return Math.abs(kernel.score(x, z));
    }
}
