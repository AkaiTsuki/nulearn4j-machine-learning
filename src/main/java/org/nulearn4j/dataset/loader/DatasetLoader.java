package org.nulearn4j.dataset.loader;

import org.nulearn4j.dataset.matrix.DoubleMatrix;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class DatasetLoader {
    public static class DoubleMatrixParser {

        public Matrix<Double> parse(File f, String delimeter) throws Exception {
            BufferedReader br = null;
            Matrix<Double> matrix = new DoubleMatrix();
            try {
                br = new BufferedReader(new FileReader(f));
                String line;
                while ((line = br.readLine()) != null) {
                    Row<Double> row = parseRow(line, delimeter);
                    if (row.size() > 0) {
                        matrix.add(row);
                    }
                }
            } catch (Exception e) {
                throw e;
            } finally {
                if (br != null) {
                    br.close();
                }
            }
            return matrix;
        }

        private Row<Double> parseRow(String line, String regex) {
            Row<Double> row = new Row<>();
            String[] splits = line.split(regex);
            for (String s : splits) {
                if (s.length() > 0) row.add(Double.parseDouble(s));
            }
            return row;
        }
    }


    public static Matrix<Double> loadBostonHousingTrain(String regex) throws Exception {
        String path = "data/housing_train.txt";
        ResourceFileReader reader = new ResourceFileReader();
        File f = reader.read(path);
        DoubleMatrixParser parser = new DoubleMatrixParser();
        return parser.parse(f, regex);
    }

    public static Matrix<Double> loadBostonHousingTest(String regex) throws Exception {
        String path = "data/housing_test.txt";
        ResourceFileReader reader = new ResourceFileReader();
        File f = reader.read(path);
        DoubleMatrixParser parser = new DoubleMatrixParser();
        return parser.parse(f, regex);
    }
}
