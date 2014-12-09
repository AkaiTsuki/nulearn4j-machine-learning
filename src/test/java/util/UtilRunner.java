package util;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by jiachiliu on 12/9/14.
 */
public class UtilRunner {

    @FunctionalInterface
    interface HeristicRule<F> {
        F measure(F f);
    }

    public static void main(String[] args) {
        double e = 0.5;
        List<Double> errors = Arrays.asList(0.2, 0.1, 0.8, 0.5, 0.7, 0.4);
        System.out.println(findMin(e, errors));
        System.out.println(findMax(e, errors));
        System.out.println(findMaxFunctional(e, errors, Math::abs));
    }

    private static int findMax(double e, List<Double> errors) {
        double maxErr = -1;
        int inx = -1;

        for (int i = 0; i < errors.size(); i++) {
            double delta = Math.abs(e - errors.get(i));
            if (delta > maxErr) {
                maxErr = delta;
                inx = i;
            }
        }
        return inx;
    }

    private static int findMaxFunctional(double e, List<Double> errors, HeristicRule<Double> heristicRule) {
        return IntStream
                .range(0, errors.size())
                .reduce(0, (partial, next) -> heristicRule.measure(e - errors.get(next)) > heristicRule.measure(e - errors.get(partial)) ? next : partial);
    }

    private static int findMin(double e, List<Double> errors) {
        double maxErr = Double.MAX_VALUE;
        int inx = -1;

        for (int i = 0; i < errors.size(); i++) {
            double delta = Math.abs(e - errors.get(i));
            if (delta < maxErr) {
                maxErr = delta;
                inx = i;
            }
        }
        return inx;
    }

}
