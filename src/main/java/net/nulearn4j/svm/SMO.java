package net.nulearn4j.svm;

import net.nulearn4j.core.classifier.Classifier;
import net.nulearn4j.svm.kernel.Kernel;
import net.nulearn4j.svm.kernel.RBFKernel;
import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.svm.kernel.LinearKernel;
import net.nulearn4j.util.MathUtil;

import java.util.*;

/**
 * Created by jiachiliu on 11/26/14.
 * <p>
 * Implement SMO for Support Vector Machine(SVM)
 */
public class SMO implements Classifier {
    private double C = 0.05;
    private double eps = 0.001;
    private double tolerance = 0.001;
    private int maxLoop = 50;

    /**
     * Linear Weights
     */
    private List<Double> ws;

    /**
     * Lagrange multipliers
     */
    private List<Double> as;

    /**
     * Threshold
     */
    private double b;

    /**
     * Cache to save error for data points
     */
    private Map<Integer, Double> errorCache = new HashMap<>();

    private Kernel kernel;

    public SMO() {
    }

    public SMO(String kernel) {
        this.kernel = getKernelInstance(kernel);
    }

    public SMO(double c, double eps, double tolerance, int maxLoop) {
        this.C = c;
        this.eps = eps;
        this.tolerance = tolerance;
        this.maxLoop = maxLoop;
        kernel = new LinearKernel();
    }

    public SMO(double c, double eps, double tolerance, int maxLoop, String kernel) {
        this(c, eps, tolerance, maxLoop);
        this.kernel = getKernelInstance(kernel);
    }

    public SMO(double c, double eps, double tolerance, int maxLoop, double gamma) {
        this(c, eps, tolerance, maxLoop);
        this.kernel = new RBFKernel(gamma);
    }

    @Override
    public List<Double> predictRawScore(Matrix<Double> test) throws Exception {
        List<Double> l = new ArrayList<>(test.getRowCount());
        for (int i = 0; i < test.getRowCount(); i++) {
            double value = predictOne(test.getRow(i).getData());
            l.add(value);
        }
        return l;
    }

    @Override
    public List<Double> predict(Matrix<Double> test) throws Exception {
        List<Double> l = new ArrayList<>(test.getRowCount());

        for (int i = 0; i < test.getRowCount(); i++) {
            double value = predictOne(test.getRow(i).getData());
            if (value <= 0.0) {
                l.add(-1.0);
            } else {
                l.add(1.0);
            }
        }

        return l;
    }

    /**
     * Given a data point, predict the raw value using weights
     *
     * @param x a data point
     * @return predict value
     * @throws Exception
     */
    public double predictOne(List<Double> x) throws Exception {
        return MathUtil.dot(ws, x) - b;
    }

    /**
     * Fit the training set to get optimization weights
     *
     * @param X training set
     * @param Y training set label
     * @throws Exception
     */
    @Override
    public Classifier fit(Matrix<Double> X, List<Double> Y) throws Exception {
        int m = X.getRowCount();
        int n = X.getColumnCount();

        int numOfChanged = 0;
        boolean examineAll = true;
        int loop = 0;

        initParameters(m, n);

        while (numOfChanged > 0 || examineAll) {
            numOfChanged = 0;
            if (examineAll) {
                for (int i = 0; i < m; i++) {
                    numOfChanged += examineExample(i, X, Y);
                }
            } else {
                for (int i = 0; i < m; i++) {
                    if (isSupportVector(as.get(i))) {
                        numOfChanged += examineExample(i, X, Y);
                    }
                }
            }

            System.out.format("iteration %3d, number of changes: %d, kernel cache miss: %f\n", loop, numOfChanged, kernel.missRate());
            if (examineAll)
                examineAll = false;
            else if (numOfChanged == 0) {
                examineAll = true;
            }
            loop++;
            if (loop > maxLoop) break;
        }
        return this;
    }

    /**
     * Heuristically choosing second Lagrange multipliers and update the chosen pair
     *
     * @param i2 index of first alpha
     * @param X  training set
     * @param Y  training set label
     * @return number of updated alphas.
     * @throws Exception
     */
    private int examineExample(int i2, Matrix<Double> X, List<Double> Y) throws Exception {
        double y2 = Y.get(i2);
        List<Double> x2 = X.getRow(i2).getData();
        double a2 = as.get(i2);
        double e2 = error(i2, X, Y);
        double r2 = e2 * y2;

        if ((r2 < -tolerance && a2 < C) || (r2 > tolerance && a2 > 0)) {
            List<Integer> indices = getSupportVectorMultipliers();

            // find the best second multiplier
            if (indices.size() > 1) {
                int i1 = findMultiplierHeuristic(e2, X, Y, indices);
                if (takeStep(i1, i2, X, Y) > 0) return 1;
            }

            // fail to find best multiplier, loop through all support vectors, start at random position
            Collections.shuffle(indices);
            for (int i = 0; i < indices.size(); i++) {
                if (takeStep(indices.get(i), i2, X, Y) > 0) return 1;
            }

            // fail to find support vectors, loop through all train data, start at random position
            List<Integer> xIndices = MathUtil.range(0, X.getRowCount());
            Collections.shuffle(xIndices);
            for (Integer i1 : xIndices) {
                if (takeStep(i1, i2, X, Y) > 0) return 1;
            }
        }
        return 0;
    }

    /**
     * Update the Lagrange Multiplier pair
     *
     * @param i1 index of alpha 1
     * @param i2 index of alpha 2
     * @param X  training set
     * @param Y  training set label
     * @return 1 if success updated else 0
     * @throws Exception
     */
    private int takeStep(int i1, int i2, Matrix<Double> X, List<Double> Y) throws Exception {
        if (i1 == i2) return 0;

        double a1 = as.get(i1);
        List<Double> x1 = X.getRow(i1).getData();
        double y1 = Y.get(i1);
        double e1 = error(i1, X, Y);

        double a2 = as.get(i2);
        List<Double> x2 = X.getRow(i2).getData();
        double y2 = Y.get(i2);
        double e2 = error(i2, X, Y);

        double s = y1 * y2;
        double L = (y1 == y2) ? Math.max(0.0, a1 + a2 - C) : Math.max(0.0, a2 - a1);
        double H = (y1 == y2) ? Math.min(C, a1 + a2) : Math.min(C, a2 - a1 + C);
        if (L == H) {
            return 0;
        }

        double k11 = kernel.transform(x1, x1, i1, i1);
        double k22 = kernel.transform(x2, x2, i2, i2);
        double k12 = kernel.transform(x1, x2, i1, i2);
        double eta = k11 + k22 - 2 * k12;
        double a2New;
        if (eta <= 0) {
            // Optional: can update or just return when eta <= 0
            double f1 = y1 * (e1 + b) - a1 * k11 - s * a2 * k12;
            double f2 = y2 * (e2 + b) - s * a1 * k12 - a2 * k22;
            double L1 = a1 + s * (a2 - L);
            double H1 = a1 + s * (a2 - H);
            double Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12;
            double Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12;
            if (Lobj < Hobj - eps)
                a2New = L;
            else if (Lobj > Hobj + eps)
                a2New = H;
            else
                a2New = a2;
//            Or just return 0 to ignore this pair
//            return 0;
        } else {
            a2New = a2 + y2 * (e1 - e2) / eta;
        }
        // chip
        if (a2New < L) a2New = L;
        else if (a2New > H) a2New = H;

        if (Math.abs(a2New - a2) < eps * (a2 + a2New + eps)) {
            return 0;
        }

        double a1New = a1 + s * (a2 - a2New);

        double b1 = b + e1 + y1 * (a1New - a1) * k11 + y2 * (a2New - a2) * k12;
        double b2 = b + e2 + y1 * (a1New - a1) * k12 + y2 * (a2New - a2) * k22;

        if (a1New < C && a1New > 0) {
            b = b1;
        } else if (a2New < C && a2New > 0) {
            b = b2;
        } else {
            b = (b1 + b2) / 2.0;
        }

        as.set(i1, a1New);
        as.set(i2, a2New);
        updateWeightsOpt(x1, y1, a1New - a1, x2, y2, a2New - a2);
        return 1;
    }

    /**
     * Compute the error of given data point, retrieve from cache if possible
     *
     * @param i index of data point
     * @param X training set
     * @param Y training set label
     * @return the error of the data point
     * @throws Exception
     */
    private double error(int i, Matrix<Double> X, List<Double> Y) throws Exception {
        if (!errorCache.containsKey(i)) {
            errorCache.put(i, predictOne(X.getRow(i).getData()) - Y.get(i));
        }
        return errorCache.get(i);
    }

    /**
     * Update weights using current optimization alpha_i and alpha_j
     *
     * @param x1      data point 1
     * @param y1      data point label 1
     * @param deltaA1 difference of new alpha_i and old alpha_i
     * @param x2      data point 2
     * @param y2      data point label 2
     * @param deltaA2 difference of new alpha_j and old alpha_j
     */
    private void updateWeightsOpt(List<Double> x1, double y1, double deltaA1, List<Double> x2, double y2, double deltaA2) {
        MathUtil.add(ws, MathUtil.multiply(x1, y1 * deltaA1));
        MathUtil.add(ws, MathUtil.multiply(x2, y2 * deltaA2));
        errorCache.clear();
    }

    /**
     * Find best alpha_j based on error.
     *
     * @param e2      error of alpha_i
     * @param X       Training set
     * @param Y       Training set labels
     * @param indices indices of candidate multipliers
     * @return index of best alpha_j
     * @throws Exception
     */
    private int findMultiplierHeuristic(double e2, Matrix<Double> X, List<Double> Y, List<Integer> indices) throws Exception {
        double maxDelta = -1.0;
        int i1 = -1;
        for (int i = 0; i < indices.size(); i++) {
            List<Double> x1 = X.getRow(i).getData();
            double y1 = Y.get(i);
            double e1 = predictOne(x1) - y1;
            double delta = Math.abs(e1 - e2);
            if (delta > maxDelta) {
                maxDelta = delta;
                i1 = i;
            }
        }
        return i1;
    }

    /**
     * @return all indices of support vectors' Lagrange multipliers
     */
    private List<Integer> getSupportVectorMultipliers() {
        List<Integer> indices = new LinkedList<>();

        for (int i = 0; i < as.size(); i++) {
            double a = as.get(i);
            if (isSupportVector(a)) {
                indices.add(i);
            }
        }
        return indices;
    }

    /**
     * @param a Lagrange multiplier
     * @return if a represents a support vector
     */
    private boolean isSupportVector(double a) {
        return a != 0.0 && a != C;
    }

    /**
     * Initialize all weights and Lagrange multiplier
     *
     * @param m number of data points
     * @param n number of features
     */
    private void initParameters(int m, int n) {
        as = MathUtil.zeros(m);
        ws = MathUtil.zeros(n);
        b = 0.0;
    }

    private Kernel getKernelInstance(String kernel) {
        switch (kernel) {
            case "linear":
                return new LinearKernel();
            case "rbf":
                return new RBFKernel();
            default:
                return new LinearKernel();
        }
    }

}
