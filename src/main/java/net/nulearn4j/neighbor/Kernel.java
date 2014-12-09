package net.nulearn4j.neighbor;

import java.util.List;

/**
 * Created by jiachiliu on 12/3/14.
 */
public interface Kernel {
    static final String EUCLIDIAN = "euclidian";
    static final String COSINE = "cosine";
    static final String GAUSSIAN = "gaussian";
    static final String POLY = "poly";

    double score(List<Double> x, List<Double> z) throws Exception;
}
