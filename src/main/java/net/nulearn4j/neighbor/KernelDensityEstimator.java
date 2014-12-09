package net.nulearn4j.neighbor;

import net.nulearn4j.core.matrix.Matrix;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 12/4/14.
 */
public class KernelDensityEstimator {

    private Map<Double, Integer> labels = new HashMap<>();
    private int m = 0;
    private Kernel kernel;

    public KernelDensityEstimator(Configuration config) {
        this.kernel = KernelFactory.getInstance(config);
    }

    public void fit(Matrix<Double> train, List<Double> targets) throws Exception {
        m = train.getRowCount();
        for (Double t : targets) {
            if (labels.containsKey(t)) {
                labels.put(t, labels.get(t) + 1);
            } else {
                labels.put(t, 1);
            }
        }
        System.out.println(labels);
    }

    public List<Double> predict(Matrix<Double> test, Matrix<Double> train, List<Double> targets) throws Exception {
        return (test.getRows().parallelStream().map(r -> predictOne(r.getData(), train, targets)).collect(Collectors.toList()));
    }

    private double probability(Double label) {
        return 1.0 * labels.get(label) / m;
    }

    private double predictOne(List<Double> z, Matrix<Double> train, List<Double> targets) {
        double[] result = new double[labels.size()];

        for (int i = 0; i < m; i++) {
            List<Double> x = train.getRow(i).getData();
            int l = targets.get(i).intValue();
            double k = 0.0;
            try {
                k = -kernel.score(z, x);
                result[l] += k;
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

        double max = -Double.MAX_VALUE;
        int label = -1;
        for (int i = 0; i < result.length; i++) {
            result[i] = probability(i * 1.0) * (result[i] / labels.get(i * 1.0));
            if (result[i] > max) {
                max = result[i];
                label = i;
            }
        }
        System.out.format("result list: %s, label: %s\n", Arrays.toString(result), label);
        return label;
    }
}
