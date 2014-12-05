package org.nulearn4j.neighbor;

import org.nulearn4j.util.Statistic.MathUtil;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 12/4/14.
 */
public class WindowKNN extends ClassificationKNN {

    public WindowKNN() {
    }

    public WindowKNN(double k) {
        super(k);
    }

    public WindowKNN(double k, Configuration config) {
        super(k, config);
    }

    @Override
    protected List<Integer> findNeighbors(List<Double> scores) {
        int m = scores.size();
        List<Integer> indices = MathUtil.range(0, m);
        return indices.stream().filter(i -> scores.get(i) <= K).collect(Collectors.toList());
    }
}
