package org.nulearn4j.util.Statistic;

import org.nulearn4j.dataset.matrix.Matrix;

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




}
