package net.nulearn4j.dataset.preprocessing.normalization;

import net.nulearn4j.dataset.matrix.Matrix;

/**
 * Created by jiachiliu on 10/18/14.
 */
public interface Normalization<T> {
    /**
     * Set up the means and standard deviations from data
     *
     * @param data a matrix data
     */
    void setUpMeanAndStd(Matrix<T> data);

    /**
     * Normalize the data in-place
     *
     * @param data the dataset that will be normalized
     */
    void normalize(Matrix<T> data);
}
