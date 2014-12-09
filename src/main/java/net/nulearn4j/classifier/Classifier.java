package net.nulearn4j.classifier;

import net.nulearn4j.dataset.matrix.Matrix;

import java.util.List;

/**
 * Created by jiachiliu on 12/1/14.
 *
 * Classifier interface that all classifiers need to implement.
 */
public interface Classifier {
    Classifier fit(Matrix<Double> train, List<Double> targets) throws Exception;

    List<Double> predict(Matrix<Double> test) throws Exception;

    List<Double> predictRawScore(Matrix<Double> test) throws Exception;
}
