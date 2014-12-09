package net.nulearn4j.linear;

import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.core.matrix.Row;
import net.nulearn4j.neighbor.Kernel;
import net.nulearn4j.util.MathUtil;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 12/5/14.
 */
public class DualPerceptron {

    private List<Double> count;
    private Kernel kernel;

    public DualPerceptron(Kernel k) {
        kernel = k;
    }

    public List<Double> weights(Matrix<Double> train) {
        int m = train.getRowCount();
        int n = train.getColumnCount();

        List<Double> w = MathUtil.zeros(n);
        for (int i = 0; i < count.size(); i++) {
            MathUtil.add(w, MathUtil.multiply(train.getRow(i).getData(), count.get(i)));
        }

        return w;
    }

    public double predictOne(List<Double> x, Matrix<Double> train) {
        double p = 0;
        try {
            for (int i = 0; i < train.getRowCount(); i++) {
                if (count.get(i) != 0)
                    p += count.get(i) * kernel.score(train.getRow(i).getData(), x);
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return p;
    }

    public List<Double> predict(Matrix<Double> test, Matrix<Double> train) {
        return test.getRows().stream().map(r -> predictOne(r.getData(), train)).collect(Collectors.toList());
    }

    public void fit(Matrix<Double> train, List<Double> target) throws Exception {
        count = MathUtil.zeros(train.getRowCount());
//        flip(train, target);

        int l = 0;
        int c;
        do {
            c = 0;
            l++;
            for (int i = 0; i < train.getRowCount(); i++) {
                Row<Double> r = train.getRow(i);
                double p = predictOne(r.getData(), train);
                if (p * target.get(i) <= 0) {
                    c += 1;
                    count.set(i, count.get(i) + target.get(i));
                }
            }
            System.out.format("iteration %d: total mistakes %d\n", l, c);
        } while (c > 0 && l < 4000);
    }

    private void flip(Matrix<Double> train, List<Double> target) {
        for (int i = 0; i < target.size(); i++) {
            if (target.get(i) < 0) {
                Row<Double> r = train.getRow(i);
                for (int j = 0; j < r.size(); j++) {
                    r.set(j, -r.get(j));
                }
                target.set(i, -target.get(i));
            }
        }
    }

}
