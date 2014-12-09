package net.nulearn4j.util.Statistic;

import java.util.List;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class DoubleListStatistic {
    /**
     * @param list a list of values
     * @return the mean of given values
     */
    public static double mean(List<Double> list) {
        return list.stream().mapToDouble(v -> v).average().getAsDouble();
    }

    /**
     * @param list a list of values
     * @return the variance of given values
     */
    public static double variance(List<Double> list) {
        double mean = mean(list);
        return variance(list, mean);
    }

    /**
     * @param list a list of values
     * @param mean the mean of given values
     * @return the variance of given values
     */
    public static double variance(List<Double> list, double mean) {
        double var = 0.0;
        for (Double val : list) {
            var += (val - mean) * (val - mean);
        }
        return var / list.size();
    }

    /**
     * @param list a list of values
     * @return the standard deviation of the given list
     */
    public static double std(List<Double> list) {
        return Math.sqrt(variance(list));
    }

    /**
     * @param list a list of values
     * @param mean the mean of given values
     * @return the standard deviation of the given list
     */
    public static double std(List<Double> list, double mean) {
        return Math.sqrt(variance(list, mean));
    }

    /**
     * @param list a list of values
     * @return the minimum value in the list
     */
    public static double min(List<Double> list) {
        return list.stream().min(Double::compare).get();
    }

    /**
     * @param list a list of values
     * @return the maximum value in the list
     */
    public static double max(List<Double> list) {
        return list.stream().max(Double::compare).get();
    }
}
