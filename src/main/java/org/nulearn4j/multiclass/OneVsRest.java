package org.nulearn4j.multiclass;

import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;
import org.nulearn4j.svm.SMO;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jiachiliu on 11/26/14.
 * A simple multi-class classifier using one-vs-others
 */
public class OneVsRest {

    List<SMO> classifiers = new ArrayList<>(10);

    public void fit(Matrix<Double> train, List<Double> trainTarget) throws Exception {
        for (double i = 0.0; i < 10.0; i++) {
            List<Double> targets = binaryLabels(trainTarget, i);
            SMO smo = new SMO(0.01, 0.001, 0.1, 200);
            System.out.format("=============== Train SMO for Label %f =============\n", i);
            smo.fit(train, targets);
            classifiers.add(smo);
        }
    }

    public List<Double> predict(Matrix<Double> test) throws Exception {
        List<Double> p = new ArrayList<>(test.getRowCount());

        for (Row<Double> r : test.getRows()) {
            double label = -1.0;
            double score = -Double.MAX_VALUE;
            for (int i = 0; i < classifiers.size(); i++) {
                double s = classifiers.get(i).predictOne(r.getData());
                if (s > score) {
                    score = s;
                    label = i;
                }
            }
            p.add(label);
        }

        return p;
    }

    private List<Double> binaryLabels(List<Double> trainTarget, double expect) {
        List<Double> l = new ArrayList<>(trainTarget.size());

        for (Double d : trainTarget) {
            if (d.equals(expect)) {
                l.add(1.0);
            } else {
                l.add(-1.0);
            }
        }
        return l;
    }
}
