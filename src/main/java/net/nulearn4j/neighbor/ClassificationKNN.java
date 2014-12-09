package net.nulearn4j.neighbor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class ClassificationKNN extends KNN {

    public ClassificationKNN() {
        super();
    }

    public ClassificationKNN(double k) {
        super(k);
    }

    public ClassificationKNN(double k, Configuration config) {
        super(k, config);
    }

    @Override
    protected double getPredictValue(List<Integer> neighbors, List<Double> trainTarget) {
        Map<Double, Integer> map = new HashMap<>();

        for (Integer i : neighbors) {
            double t = trainTarget.get(i);
            if (map.containsKey(t)) {
                map.put(t, map.get(t) + 1);
            } else {
                map.put(t, 1);
            }
        }

        int max = 0;
        double major = Double.NaN;

        for (Map.Entry<Double, Integer> entry : map.entrySet()) {
            if (entry.getValue() > max) {
                major = entry.getKey();
                max = entry.getValue();
            }
        }

        return major;
    }
}
