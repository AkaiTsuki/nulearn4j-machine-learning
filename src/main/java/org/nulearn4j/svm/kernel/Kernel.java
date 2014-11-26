package org.nulearn4j.svm.kernel;

import java.util.List;

/**
 * Created by jiachiliu on 11/26/14.
 */
public interface Kernel {
    public double transform(List<Double> x1, List<Double> x2, int i1, int i2) throws Exception;

    public double missRate();
}
