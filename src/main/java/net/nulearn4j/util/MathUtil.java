package net.nulearn4j.util;

import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.core.matrix.Row;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by jiachiliu on 10/19/14.
 */
public class MathUtil {
    /**
     * @param val a value
     * @return the log value based on 2
     */
    public static double log2(double val) {
        return Math.log(val) / Math.log(2);
    }

    public static double det(Matrix<Double> matrix) {
        double[][] array2d = matrix.toPrimitive();
        Jama.Matrix m = new Jama.Matrix(array2d);
        return m.det();
    }

    /**
     * Add the second vector to the first one
     *
     * @param vector
     * @param toAdd
     */
    public static void add(List<Double> vector, List<Double> toAdd) {
        for (int i = 0; i < vector.size(); i++) {
            double sum = vector.get(i) + toAdd.get(i);
            vector.set(i, sum);
        }
    }

    public static List<Double> minus(List<Double> a, List<Double> b) throws Exception {
        List<Double> l = new ArrayList<>(a.size());

        for (int i = 0; i < a.size(); i++) {
            l.add(a.get(i) - b.get(i));
        }

        return l;
    }

    public static double distance(List<Double> a, List<Double> b) throws Exception {
        double d = 0;
        for (int i = 0; i < a.size(); i++) {
            d += (a.get(i) - b.get(i)) * (a.get(i) - b.get(i));
        }
        return Math.sqrt(d);
    }

    public static List<Double> ones(int m) {
        return ns(m, 1.0);
    }

    public static List<Double> zeros(int m) {
        return ns(m, 0.0);
    }

    public static List<Double> ns(int m, double val) {
        List<Double> l = new ArrayList<>(m);
        for (int i = 0; i < m; i++) {
            l.add(val);
        }
        return l;
    }

    public static List<Double> rand(int m) {
        Random rand = new Random();
        List<Double> l = new ArrayList<>(m);
        for (int i = 0; i < m; i++) {
            l.add(rand.nextDouble());
        }
        return l;
    }

    public static List<Integer> range(int s, int e) {
        List<Integer> l = new ArrayList<>(e - s);

        for (int i = s; i < e; i++) {
            l.add(i);
        }
        return l;
    }

    public static List<Double> multiply(List<Double> vector, double scalar) {
        List<Double> r = new ArrayList<>(vector.size());

        for (Double aVector : vector) {
            r.add(aVector * scalar);
        }

        return r;
    }

    public static List<Integer> argsort(List<Double> f, boolean reverse) {
        int m = f.size();
        List<Integer> args = new ArrayList<>(m);
        for (int i = 0; i < m; i++) {
            args.add(i);
        }
        if (reverse) {
            Collections.sort(args, (a, b) -> -f.get(a).compareTo(f.get(b)));
        } else {
            Collections.sort(args, (a, b) -> f.get(a).compareTo(f.get(b)));
        }
        return args;
    }

    public static double dot(Row<Double> r1, Row<Double> r2) throws Exception {
        return dot(r1.getData(), r2.getData());
    }

    public static double dot(List<Double> r1, List<Double> r2) throws Exception {
        if (r1.size() != r2.size()) {
            throw new Exception("Dimension disagree");
        }

        double sum = 0.0;
        for (int i = 0; i < r1.size(); i++) {
            sum += r1.get(i) * r2.get(i);
        }
        return sum;
    }


}
