package net.nulearn4j.multiclass;

import net.nulearn4j.svm.SMO;
import net.nulearn4j.core.matrix.Matrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jiachiliu on 11/27/14.
 */
public class ECOC {

    private List<SMO> smos = new LinkedList<>();
    private int numOfLabels = 10;
    private int numOfFunc = 50;

    public double[][] fit(Matrix<Double> train, List<Double> trainTarget) throws Exception {
        double[][] code = randomSelect(exhaustiveCode(numOfLabels), numOfFunc);
        for (int f = 0; f < code[0].length; f++) {
            List<Double> target = binaryLabels(trainTarget, code, f);
            System.out.format("============== Train function %d=============\n", f);
            SMO smo = new SMO(0.01, 0.001, 0.1, 400);
            smo.fit(train, target);
            smos.add(smo);
        }
        return code;
    }

    public List<Double> predict(Matrix<Double> test, double[][] code) throws Exception {
        List<Double> p = new ArrayList<>(test.getRowCount());

        for (int t = 0; t < test.getRowCount(); t++) {
            List<Double> r = new LinkedList<>();
            for (SMO smo : smos) {
                double score = smo.predictOne(test.getRow(t).getData());
                if (score <= 0) {
                    r.add(0.0);
                } else {
                    r.add(1.0);
                }
            }
            p.add(getLabel(r, code));
        }
        System.out.println(p);
        return p;
    }

    private double getLabel(List<Double> v, double[][] code) {
        int distance = Integer.MAX_VALUE;
        int label = -1;


        for (int i = 0; i < code.length; i++) {
            int d = 0;
            for (int j = 0; j < v.size(); j++) {
                if (!v.get(j).equals(code[i][j])) {
                    d++;
                }
            }
            if (d < distance) {
                distance = d;
                label = i;
            }
        }

        return label;
    }

    private List<Double> binaryLabels(List<Double> target, double[][] code, int col) {
        List<Double> l = new ArrayList<>(target.size());
        for (Double t : target) {
            if (code[t.intValue()][col] == 0.0) {
                l.add(-1.0);
            } else {
                l.add(1.0);
            }
        }
        return l;
    }


    public double[][] randomSelect(double[][] code, int numOfCols) {
        double[][] sample = new double[code.length][numOfCols];
        List<Integer> cols = new LinkedList<>();
        for (int i = 0; i < code[0].length; i++) {
            cols.add(i);
        }
        Collections.shuffle(cols);
        for (int r = 0; r < code.length; r++) {
            int col = 0;
            for (int i = 0; i < numOfCols; i++) {
                sample[r][col++] = code[r][cols.get(i)];
            }
        }
        return sample;
    }

    public double[][] exhaustiveCode(int numOfLabels) {
        int k = numOfLabels;
        int n = (int) Math.pow(2.0, (k - 1)) - 1;
        double[][] code = new double[k][n];

        for (int i = 0; i < n; i++)
            code[0][i] = 1.0;

        for (int i = 1; i < k; i++) {
            int interval = (int) Math.pow(2, (k - i - 1));
            int count = 0;
            boolean flag = false;
            for (int c = 0; c < n; c++) {
                code[i][c] = flag ? 1.0 : 0.0;
                count++;
                if (count == interval) {
                    flag = !flag;
                    count = 0;
                }
            }
        }
        return code;
    }

    public static void print2DArray(double[][] matrix) {
        System.out.println("==============Matrix============");
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.print(" " + matrix[i][j]);
            }
            System.out.println();
        }
        System.out.println("==============Matrix============");
    }
}
