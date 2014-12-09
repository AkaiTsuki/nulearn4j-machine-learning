package net.nulearn4j.neighbor;

import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.util.MathUtil;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 12/3/14.
 * Abstract class control the KNN process
 */
public abstract class KNN {

    protected double K = 10;
    private Kernel kernel = new EuclidianDistanceKernel();

    public KNN() {
    }

    public KNN(double k) {
        K = k;
    }

    public KNN(double k, Configuration config) {
        K = k;
        this.kernel = KernelFactory.getInstance(config);
    }


    public List<Double> predict(Matrix<Double> train, List<Double> trainTarget, Matrix<Double> test) throws Exception {
        return test.getRows().parallelStream().map(t -> predictOne(train, trainTarget, t.getData())).collect(Collectors.toList());
    }

    private double predictOne(Matrix<Double> train, List<Double> trainTarget, List<Double> p) {
        List<Double> scores = train.getRows().parallelStream().map(t -> calculateScore(t.getData(), p)).collect(Collectors.toList());

        List<Integer> neighbors = findNeighbors(scores);
        return getPredictValue(neighbors, trainTarget);
    }

    protected List<Integer> findNeighbors(List<Double> scores) {
        int m = scores.size();
        List<Integer> indices = MathUtil.range(0, m);
        Collections.sort(indices, (a, b) -> scores.get(a).compareTo(scores.get(b)));
        return indices.subList(0, (int) K);
    }

    protected double calculateScore(List<Double> data, List<Double> p) {
        try {
            return kernel.score(data, p);
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return 0.0;
        }
    }

    protected abstract double getPredictValue(List<Integer> neighbors, List<Double> trainTarget);

    @Override
    public String toString() {
        return "KNN{" +
                "K=" + K +
                ", kernel=" + kernel +
                '}';
    }
}
