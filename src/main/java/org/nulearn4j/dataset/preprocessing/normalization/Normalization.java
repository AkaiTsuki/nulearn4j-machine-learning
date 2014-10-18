package org.nulearn4j.dataset.preprocessing.normalization;

import org.nulearn4j.dataset.matrix.Matrix;

import java.util.List;

/**
 * Created by jiachiliu on 10/18/14.
 */
public interface Normalization<T> {
    void setUpMeanAndVariance(Matrix<T> data);
    void normalize(Matrix<T> data);
}
