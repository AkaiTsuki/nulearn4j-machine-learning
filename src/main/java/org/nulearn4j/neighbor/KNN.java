package org.nulearn4j.neighbor;

import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.util.Statistic.MathUtil;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 12/3/14.
 * Abstract class control the KNN process
 */
public abstract class KNN {

    private int K = 10;
    private Kernel kernel = new EuclidianDistanceKernel();

    public KNN() {
    }

    public KNN(int k) {
        K = k;
    }

    public KNN(int k, String kernel) {
        K = k;
        this.kernel = getKernel(kernel);
    }

    protected Kernel getKernel(String kernel) {
        switch (kernel) {
            case Kernel.EUCLIDIAN:
                return new EuclidianDistanceKernel();
            case Kernel.COSINE:
                return new CosineKernel();
            case Kernel.GAUSSIAN:
                return new GaussianKernel();
            case Kernel.POLY:
                return new PolyKernel();
            default:
                return new EuclidianDistanceKernel();
        }
    }

    public List<Double> predict(Matrix<Double> train, List<Double> trainTarget, Matrix<Double> test) throws Exception {
        return test.getRows().parallelStream().map(t -> predictOne(train, trainTarget, t.getData())).collect(Collectors.toList());
    }

    private double predictOne(Matrix<Double> train, List<Double> trainTarget, List<Double> p) {
        int m = train.getRowCount();
        List<Integer> indices = MathUtil.range(0, m);

        List<Double> scores = train.getRows().parallelStream().map(t -> calculateScore(t.getData(), p)).collect(Collectors.toList());
        Collections.sort(indices, (a, b) -> scores.get(a).compareTo(scores.get(b)));

//        List<Integer> neighbors = indices.subList(indices.size() - K, indices.size());
        List<Integer> neighbors = indices.subList(0, K);
        return getPredictValue(neighbors, trainTarget);
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
