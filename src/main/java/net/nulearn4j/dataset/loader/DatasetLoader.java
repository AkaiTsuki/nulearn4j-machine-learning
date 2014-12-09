package net.nulearn4j.dataset.loader;

import net.nulearn4j.core.matrix.DoubleMatrix;
import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.core.matrix.Row;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class DatasetLoader {
    public static class DoubleMatrixParser {

        public Matrix<Double> parse(File f, String delimiter) throws Exception {
            BufferedReader br = null;
            Matrix<Double> matrix = new DoubleMatrix();
            try {
                br = new BufferedReader(new FileReader(f));
                String line;
                while ((line = br.readLine()) != null) {
                    Row<Double> row = parseRow(line, delimiter);
                    if (row.size() > 0) {
                        matrix.add(row);
                    }
                }
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

    public static List<Double> loadLabel(String path) throws Exception {
        ResourceFileReader reader = new ResourceFileReader();
        File f = reader.read(path);
        List<Double> targets = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(f));
        String line;

        while ((line = br.readLine()) != null) {
            if (line.trim().length() > 0)
                targets.add(Double.parseDouble(line.trim()));
        }
        return targets;
    }

    public static Matrix<Double> loadData(String regex, String path) throws Exception {
        ResourceFileReader reader = new ResourceFileReader();
        File f = reader.read(path);
        DoubleMatrixParser parser = new DoubleMatrixParser();
        return parser.parse(f, regex);
    }

    public static Matrix<Double> loadBostonHousingTrain(String regex) throws Exception {
        String path = "data/housing_train.txt";
        return loadData(regex, path);
    }

    public static Matrix<Double> loadBostonHousingTest(String regex) throws Exception {
        String path = "data/housing_test.txt";
        return loadData(regex, path);
    }

    public static Matrix<Double> loadSpambase(String regex) throws Exception {
        String path = "data/spambase.data";
        return loadData(regex, path);
    }

    public static Matrix<Double> loadPerceptronData(String regex) throws Exception {
        String path = "data/perceptronData.txt";
        return loadData(regex, path);
    }

    public static Matrix<Double> loadTwoSpiralData(String regex) throws Exception {
        String path = "data/twoSpirals.txt";
        return loadData(regex, path);
    }

    public static Matrix<Double> loadPolluteSpamBaseTrain(String regex) throws Exception {
        String path = "data/spam_polluted/train_feature.txt";
        return loadData(regex, path);
    }

    public static Matrix<Double> loadPolluteSpamBaseTest(String regex) throws Exception {
        String path = "data/spam_polluted/test_feature.txt";
        return loadData(regex, path);
    }

}
