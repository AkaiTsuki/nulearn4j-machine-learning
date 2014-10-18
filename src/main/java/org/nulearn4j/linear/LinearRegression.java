package org.nulearn4j.linear;

import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class LinearRegression {
    public static void main(String[] args) throws Exception{
        DatasetLoader loader = new DatasetLoader();
        Matrix<Double> train = loader.loadBostonHousingTrain("\\s+");
        Matrix<Double> test = loader.loadBostonHousingTest("\\s+");
    }
}
