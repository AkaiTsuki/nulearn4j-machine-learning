package neighbor;

import net.nulearn4j.dataset.loader.DatasetLoader;
import net.nulearn4j.core.matrix.Matrix;
import net.nulearn4j.normalization.Normalization;
import net.nulearn4j.normalization.ZeroMeanUnitVar;
import net.nulearn4j.linear.DualPerceptron;
import net.nulearn4j.neighbor.*;
import net.nulearn4j.validation.Validation;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class KNNRunner {

    public static void twoSpiral() throws Exception {
        Matrix<Double> train = DatasetLoader.loadTwoSpiralData("\\s+");
        train.shuffle();

        Matrix<Double>[] splits = train.split(train.getRowCount() / 10);
        Matrix<Double> test = splits[0];
        train = splits[1];

        int label = train.getColumnCount() - 1;
        List<Double> trainTarget = train.getColumn(label);
        List<Double> testTarget = test.getColumn(label);
        train = train.removeColumn(label);
        test = test.removeColumn(label);

        train = train.addColumn(0, 1.0);
        test = test.addColumn(0, 1.0);

        DualPerceptron perceptron = new DualPerceptron(new AbsoluteKernel(new GaussianKernel(5)));
//        DualPerceptron perceptron = new DualPerceptron(new DotProductKernel());
        perceptron.fit(train, trainTarget);
        List<Double> predicts = perceptron.predict(test, train);
//        System.out.println(predicts);
//        System.out.println(testTarget);
        predicts = predicts.stream().map(p -> (p <= 0.5) ? -1.0 : 1.0).collect(Collectors.toList());
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predicts, testTarget);
        System.out.println("============== Test Performance==========\n" + cm);
    }

    public static void dualPerceptron() throws Exception {
        Matrix<Double> train = DatasetLoader.loadPerceptronData("\t");
        train = train.addColumn(0, 1.0);

        int label = train.getColumnCount() - 1;
        List<Double> trainTarget = train.getColumn(label);
        train = train.removeColumn(label);

        DualPerceptron perceptron = new DualPerceptron(new DotProductKernel());
        perceptron.fit(train, trainTarget);
    }

    public static void digitalKDE(int size, Configuration config) throws Exception {
        System.out.format("Load Dataset...\n");
        Matrix<Double> train = DatasetLoader.loadData(",", "data/digital_train_features.txt");
        Matrix<Double> test = DatasetLoader.loadData(",", "data/digital_test_features.txt");
        List<Double> trainTarget = DatasetLoader.loadLabel("data/digital_train_target.txt");
        List<Double> testTarget = DatasetLoader.loadLabel("data/digital_test_target.txt");

        Normalization<Double> norm = new ZeroMeanUnitVar();
        norm.setUpMeanAndStd(train);
        norm.normalize(train);
        norm.normalize(test);

        train = train.split(size)[0];
        trainTarget = trainTarget.subList(0, size);

        int[] counts = new int[10];
        for (int i = 0; i < train.getRowCount(); i++) {
            counts[trainTarget.get(i).intValue()] += 1;
        }
        System.out.println("Class statistic: " + Arrays.toString(counts));

        KernelDensityEstimator clf = new KernelDensityEstimator(config);
        System.out.println(clf);
        clf.fit(train, trainTarget);
        System.out.println(clf);
        List<Double> predicts = clf.predict(test, train, trainTarget);

        double error = 0.0;
        for (int i = 0; i < predicts.size(); i++) {
            if (!predicts.get(i).equals(testTarget.get(i))) {
                error += 1.0;
            }
        }
        System.out.format("Total Acc: %f, Total Errors: %f\n", (1 - error / predicts.size()), error / predicts.size());
    }

    public static void spambaseKDE(Configuration config) throws Exception {
        Matrix<Double> spambase = DatasetLoader.loadSpambase(",");
        spambase.shuffle();

        Matrix<Double>[] splits = spambase.split(spambase.getRowCount() / 10);
        Matrix<Double> test = splits[0];
        Matrix<Double> train = splits[1];

        int label = train.getColumnCount() - 1;
        List<Double> trainTarget = train.getColumn(label);
        List<Double> testTarget = test.getColumn(label);
        train = train.removeColumn(label);
        test = test.removeColumn(label);

        Normalization<Double> normalization = new ZeroMeanUnitVar();
        normalization.setUpMeanAndStd(train);
        normalization.normalize(train);
        normalization.normalize(test);

        KernelDensityEstimator clf = new KernelDensityEstimator(config);
        System.out.println(clf);
        clf.fit(train, trainTarget);
        List<Double> predicts = clf.predict(test, train, trainTarget);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predicts, testTarget);
        System.out.println("============== Test Performance==========\n" + cm);
    }

    public static void spambase(double k, Configuration config) throws Exception {
        Matrix<Double> spambase = DatasetLoader.loadSpambase(",");
        spambase.shuffle();

        Matrix<Double>[] splits = spambase.split(spambase.getRowCount() / 10);
        Matrix<Double> test = splits[0];
        Matrix<Double> train = splits[1];

        int label = train.getColumnCount() - 1;
        List<Double> trainTarget = train.getColumn(label);
        List<Double> testTarget = test.getColumn(label);
        train = train.removeColumn(label);
        test = test.removeColumn(label);

        Normalization<Double> normalization = new ZeroMeanUnitVar();
        normalization.setUpMeanAndStd(train);
        normalization.normalize(train);
        normalization.normalize(test);

        KNN clf;
        if (config.get("type").equals("window")) {
            clf = new WindowKNN(k, config);
        } else {
            clf = new ClassificationKNN(k, config);
        }
        System.out.println(clf);

        List<Double> predicts = clf.predict(train, trainTarget, test);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predicts, testTarget);
        System.out.println("============== Test Performance==========\n" + cm);
    }

    public static void digital(int size, double k, Configuration config) throws Exception {
        System.out.format("Load Dataset...\n");
        Matrix<Double> train = DatasetLoader.loadData(",", "data/digital_train_features.txt");
        Matrix<Double> test = DatasetLoader.loadData(",", "data/digital_test_features.txt");
        List<Double> trainTarget = DatasetLoader.loadLabel("data/digital_train_target.txt");
        List<Double> testTarget = DatasetLoader.loadLabel("data/digital_test_target.txt");

        Normalization<Double> norm = new ZeroMeanUnitVar();
        norm.setUpMeanAndStd(train);
        norm.normalize(train);
        norm.normalize(test);

        train = train.split(size)[0];
        trainTarget = trainTarget.subList(0, size);

        int[] counts = new int[10];
        for (int i = 0; i < train.getRowCount(); i++) {
            counts[trainTarget.get(i).intValue()] += 1;
        }
        System.out.println("Class statistic: " + Arrays.toString(counts));

        KNN clf;
        if (config.get("type").equals("window")) {
            clf = new WindowKNN(k, config);
        } else {
            clf = new ClassificationKNN(k, config);
        }
        System.out.println(clf);
        List<Double> predicts = clf.predict(train, trainTarget, test);

        double error = 0.0;
        for (int i = 0; i < predicts.size(); i++) {
            if (!predicts.get(i).equals(testTarget.get(i))) {
                error += 1.0;
            }
        }
        System.out.format("Total Acc: %f, Total Errors: %f\n", (1 - error / predicts.size()), error / predicts.size());
    }

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();

        Configuration config = new Configuration();

        /* PB1-1 Spambase */
//        config.set("type", "normal");
//        config.set("kernel", Kernel.EUCLIDIAN);
//        spambase(7, config);

        /* PB1-1 digital */
//        config.set("type", "normal");
//        config.set("kernel", Kernel.GAUSSIAN);

//        config.setDouble("degree", 2.0);
//        config.setDouble("C", 20.0);
//        digital(12000, 1, config);

        /* PB2-1 Spambase */
//        config.set("type", "window");
//        config.set("kernel", Kernel.EUCLIDIAN);
//        spambase(4, config);

        /* PB2-1 Digital */
//        config.set("type", "window");
//        config.set("kernel", Kernel.COSINE);
//        digital(12000, -0.8, config);


        /* PB2-2 spambase with KDE */
//        config.set("kernel", Kernel.GAUSSIAN);
//        config.setDouble("gamma", 1.0);
//        spambaseKDE(config);

        /* PB2-2 digital with KDE */
//        config.set("kernel", Kernel.GAUSSIAN);
//        config.setDouble("gamma", 1.0);
//        digitalKDE(12000, config);

        /* PB2-2 digital with KDE */
//        config.set("kernel", Kernel.POLY);
//        config.setDouble("degree", 2.0);
//        config.setDouble("C", 15.0);
//        digitalKDE(12000, config);

//        dualPerceptron();
        twoSpiral();

        final long endTime = System.nanoTime();
        System.out.format("Total Run time: %f secs\n", 1.0 * (endTime - startTime) / 1e9);
    }
}
