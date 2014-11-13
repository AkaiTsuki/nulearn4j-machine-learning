package org.nulearn4j.util.Statistic;

import org.nulearn4j.dataset.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jiachiliu on 10/19/14.
 */
public class MathUtil {
    /**
     * @param val a value
     * @return the log value based on 2
     */
    public static double log2(double val) {
        return Math.log(val) / Math.log(2);
    }

    public static double det(Matrix<Double> matrix) {
        double[][] array2d = matrix.toPrimitive();
        Jama.Matrix m = new Jama.Matrix(array2d);
        return m.det();
    }

    /**
     * Add the second vector to the first one
     *
     * @param vector
     * @param toAdd
     */
    public static void add(List<Double> vector, List<Double> toAdd) {
        for (int i = 0; i < vector.size(); i++) {
            double sum = vector.get(i) + toAdd.get(i);
            vector.set(i, sum);
        }
    }

    public static List<Double> ones(int m) {
        return ns(m, 1.0);
    }

    public static List<Double> zeros(int m) {
        return ns(m, 0.0);
    }

    public static List<Double> ns(int m, double val) {
        List<Double> l = new ArrayList<>(m);
        for (int i = 0; i < m; i++) {
            l.add(val);
        }
        return l;
    }

    public static List<Double> multiply(List<Double> vector, double scalar) {
        List<Double> r = new ArrayList<>(vector.size());

        for (Double aVector : vector) {
            r.add(aVector * scalar);
        }

        return r;
    }


}
