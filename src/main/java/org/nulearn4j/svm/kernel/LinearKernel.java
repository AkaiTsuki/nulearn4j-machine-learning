package org.nulearn4j.svm.kernel;

import org.nulearn4j.util.Statistic.MathUtil;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by jiachiliu on 11/26/14.
 */
public class LinearKernel implements Kernel {
    protected Map<Integer, Map<Integer, Double>> cache;
    protected int totalAccess;
    protected int miss;

    public LinearKernel() {
        cache = new HashMap<>();
    }

    public double transform(List<Double> x1, List<Double> x2, int i1, int i2) throws Exception {
        if (!cache.containsKey(i1)) {
            cache.put(i1, new HashMap<Integer, Double>());
            cache.get(i1).put(i2, dotProduct(x1, x2));
            miss++;
        } else if (!cache.get(i1).containsKey(i2)) {
            cache.get(i1).put(i2, dotProduct(x1, x2));
            miss++;
        }
        totalAccess += 1;
        return cache.get(i1).get(i2);
    }

    protected double dotProduct(List<Double> x1, List<Double> x2) throws Exception {
        return MathUtil.dot(x1, x2);
    }

    public double missRate() {
        return 1.0 * miss / totalAccess;
    }
}
