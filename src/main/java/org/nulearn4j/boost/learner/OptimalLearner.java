package org.nulearn4j.boost.learner;

import org.nulearn4j.boost.AdaBoost;
import org.nulearn4j.dataset.matrix.Matrix;

import java.util.*;
import java.util.stream.Stream;

/**
 * Created by jiachiliu on 11/11/14.
 */
public class OptimalLearner {

    private int feature = -1;
    private double threshold = Double.NaN;
    private double weightedError = -1;
    private Map<Integer, List<Cache>> cache = new HashMap<>();


    public void fit(Matrix<Double> train, List<Double> target, List<Double> weights) {
        int n = train.getColumnCount();
        feature = -1;
        threshold = Double.NaN;
        weightedError = -1.0;
        double maxErr = -1;

        for (int f = 0; f < n; f++) {
            ErrorSet r = null;
            if (cache.containsKey(f)) {
                r = this.findBestErrorOnCache(cache.get(f), weights);
            } else {
                r = this.findBestErrorOnFeature(train.getColumn(f), target, weights, f);
            }
            if (r.error > maxErr) {
                maxErr = r.error;
                this.feature = f;
                this.weightedError = r.weightedError;
                this.threshold = r.threshold;
            }
        }
    }

    private ErrorSet findBestErrorOnCache(List<Cache> caches, List<Double> w) {
        ErrorSet r = new ErrorSet();

        double currentWeightedErr = 0.0;
        for(Cache c: caches){
            List<Integer> toAdd = c.toAdd;
            List<Integer> toRemove = c.toRemove;
            double currentThreshold = c.threshold;

            for(Integer a: toAdd){
                currentWeightedErr += w.get(a);
            }

            for(Integer a: toRemove){
                currentWeightedErr -= w.get(a);
            }

            double currentAbsError = absErr(currentWeightedErr);
            if(currentAbsError > r.error){
                r.error = currentAbsError;
                r.weightedError = currentWeightedErr;
                r.threshold = currentThreshold;
            }
        }
        return r;
    }

    private ErrorSet findBestErrorOnFeature(List<Double> f, List<Double> t, List<Double> d, int index) {
        int m = d.size();

        List<Integer> args = getSortedIndices(f);
        List<Double> p = getInitPredicts(m);
        List<Integer> mismatch = mismatch(p, t);

        double currentWeightedErr = 0.0;
        for (Integer i : mismatch) {
            currentWeightedErr += d.get(i);
        }
        double currentAbsErr = absErr(currentWeightedErr);
        double currentThreshold = f.get(args.get(0)) - 0.5;

        Cache c = new Cache(mismatch, new ArrayList<Integer>(), currentThreshold);
        List<Cache> lc = new LinkedList<>();
        lc.add(c);
        this.cache.put(index, lc);

        ErrorSet r = new ErrorSet();
        r.error = currentAbsErr;
        r.weightedError = currentWeightedErr;
        r.threshold = currentThreshold;

        int start = 0;
        for (int i = 1; i < m + 1; i++) {
            if (i == m || !f.get(args.get(i)).equals(f.get(args.get(i - 1)))) {
                List<Integer> toChange = new ArrayList<>();
                List<Integer> toAdd = new LinkedList<>();
                List<Integer> toRemove = new LinkedList<>();
                for (int j = start; j < i; j++) {
                    toChange.add(args.get(j));
                }
                double delta = update(p, t, d, toChange, toAdd, toRemove);
                currentWeightedErr += delta;
                currentAbsErr = absErr(currentWeightedErr);
                currentThreshold = (i == m) ? f.get(args.get(m - 1)) + 0.5 : (f.get(args.get(i - 1)) + f.get(args.get(i))) / 2.0;
                Cache currentCache = new Cache(toAdd, toRemove, currentThreshold);
                this.cache.get(index).add(currentCache);
                if (currentAbsErr > r.error) {
                    r.error = currentAbsErr;
                    r.weightedError = currentWeightedErr;
                    r.threshold = currentThreshold;
                }
                start = i;
            }
        }

        return r;
    }

    private double update(List<Double> p, List<Double> t, List<Double> d, List<Integer> toChange, List<Integer> toAdd, List<Integer> toRemove) {
        double delta = 0.0;
        for (Integer i : toChange) {
            p.set(i, -1.0);
            if (p.get(i).equals(t.get(i))) {
                toRemove.add(i);
                delta -= d.get(i);
            } else {
                toAdd.add(i);
                delta += d.get(i);
            }
        }
        return delta;
    }

    private double absErr(double err) {
        return Math.abs(0.5 - err);
    }

    private List<Integer> mismatch(List<Double> p, List<Double> t) {
        List<Integer> m = new ArrayList<>();

        for (int i = 0; i < p.size(); i++) {
            if (!p.get(i).equals(t.get(i)))
                m.add(i);
        }

        return m;
    }

    private List<Integer> getSortedIndices(List<Double> f) {
        // argsort
        int m = f.size();
        List<Integer> args = new ArrayList<>(m);
        for (int i = 0; i < m; i++) {
            args.add(i);
        }
        Collections.sort(args, (a, b) -> f.get(a).compareTo(f.get(b)));
        return args;
    }

    private List<Double> getInitPredicts(int m) {
        List<Double> p = new ArrayList<>(m);

        for (int i = 0; i < m; i++) {
            p.add(1.0);
        }

        return p;
    }

    public int getFeature() {
        return feature;
    }

    public void setFeature(int feature) {
        this.feature = feature;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public double getWeightedError() {
        return weightedError;
    }

    public void setWeightedError(double weightedError) {
        this.weightedError = weightedError;
    }

    public static class ErrorSet {
        double error = -1.0;
        double weightedError = -1.0;
        double threshold = Double.NaN;

        @Override
        public String toString() {
            return "ErrorSet{" +
                    "error=" + error +
                    ", weightedError=" + weightedError +
                    ", threshold=" + threshold +
                    '}';
        }
    }

    public static class Cache {
        List<Integer> toAdd;
        List<Integer> toRemove;
        double threshold;

        public Cache(List<Integer> toAdd, List<Integer> toRemove, double threshold) {
            this.toAdd = toAdd;
            this.toRemove = toRemove;
            this.threshold = threshold;
        }
    }
}
