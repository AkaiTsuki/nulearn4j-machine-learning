package org.nulearn4j.multiclass;

import org.nulearn4j.classifier.Classifier;
import org.nulearn4j.dataset.matrix.DoubleMatrix;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;
import org.nulearn4j.svm.SMO;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jiachiliu on 12/1/14.
 */
public class OneVsOne {
    SMO[][] classifiers = new SMO[10][10];

    public void fit(Matrix<Double> train, List<Double> trainTarget) throws Exception {
        for (int i = 0; i < 10; i++) {
            for (int j = i + 1; j < 10; j++) {
                System.out.format("=========== Train %d vs %d =========\n", i, j);
                Matrix<Double> sample = new DoubleMatrix();
                List<Double> sampleLabels = new ArrayList<Double>();
                for (int t = 0; t < train.getRowCount(); t++) {
                    if (trainTarget.get(t) == i || trainTarget.get(t) == j) {
                        sample.add(train.getRow(t));
                        sampleLabels.add(trainTarget.get(t) == i ? 1.0 : -1.0);
                    }
                }
                SMO smo = new SMO(0.01, 0.001, 0.1, 200);
                if (sample.getRowCount() == 0) {
                    throw new Exception("Sample row count is 0");
                }
                smo.fit(sample, sampleLabels);
                classifiers[i][j] = smo;
            }
        }
    }

    public List<Double> predict(Matrix<Double> test) throws Exception {
        List<Double> p = new ArrayList<>(test.getRowCount());

        for (Row<Double> t : test.getRows()) {
            int[] counts = new int[10];
            for (int i = 0; i < 10; i++) {
                for (int j = i + 1; j < 10; j++) {
                    double val = classifiers[i][j].predictOne(t.getData());
                    if (val > 0) {
                        counts[i]++;
                    } else {
                        counts[j]++;
                    }
                }
            }
            int max = counts[0];
            int label = 0;
            for (int k = 1; k < 10; k++) {
                if (counts[k] > max) {
                    label = k;
                    max = counts[k];
                }
            }
            p.add(1.0 * label);
        }
        return p;
    }
}
