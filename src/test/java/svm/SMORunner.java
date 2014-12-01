package svm;

import org.nulearn4j.classifier.Classifier;
import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.preprocessing.normalization.Normalization;
import org.nulearn4j.dataset.preprocessing.normalization.ZeroMeanUnitVar;
import org.nulearn4j.multiclass.ECOC;
import org.nulearn4j.multiclass.OneVsOne;
import org.nulearn4j.multiclass.OneVsRest;
import org.nulearn4j.svm.SMO;
import org.nulearn4j.validation.Validation;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


public class SMORunner {

    public static void ecoc(int size) throws Exception {
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


        ECOC ecoc = new ECOC();

        double[][] code = ecoc.fit(train, trainTarget);
        List<Double> predicts = ecoc.predict(test, code);
        double error = 0.0;
        for (int i = 0; i < predicts.size(); i++) {
            if (!predicts.get(i).equals(testTarget.get(i))) {
                error += 1.0;
            }
        }
        System.out.format("Total Acc: %f, Total Errors: %f", (1 - error / predicts.size()), error / predicts.size());
    }

    public static void digital(int size) throws Exception {
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

        OneVsOne svc = new OneVsOne();
        svc.fit(train, trainTarget);
        List<Double> predicts = svc.predict(test);

        double error = 0.0;
        for (int i = 0; i < predicts.size(); i++) {
            if (!predicts.get(i).equals(testTarget.get(i))) {
                error += 1.0;
            }
        }
        System.out.format("Total Acc: %f, Total Errors: %f", (1 - error / predicts.size()), error / predicts.size());
    }

    public static void kfoldSpambase(double c, double eps, double tol, int max) throws Exception {
        Matrix<Double> spambase = DatasetLoader.loadSpambase(",");
        // Shuffle the data since it put all spam at the head fo file.
        spambase.shuffle();

        int k = 10;

        double accuracy = 0.0;
        double error = 0.0;

        for (int i = 0; i < k; i++) {
            System.out.println("\n============== Fold " + i + "=================");
            Matrix<Double> train = spambase.kFoldTrain(k, i);
            Matrix<Double> test = spambase.kFoldTest(k, i);
            Validation.ConfusionMatrix cm = run(test, train, c, eps, tol, max);
            accuracy += cm.accuracy();
            error += cm.error();
        }

        System.out.println("Average Accuracy: " + accuracy / k + " Average error: " + error / k);
    }

    private static Validation.ConfusionMatrix run(Matrix<Double> test, Matrix<Double> train,
                                                  double c, double eps, double tol, int max) throws Exception {
        int label = train.getColumnCount() - 1;
        List<Double> trainTarget = train.getColumn(label);
        List<Double> testTarget = test.getColumn(label);

        train = train.removeColumn(label);
        test = test.removeColumn(label);

        Normalization<Double> norm = new ZeroMeanUnitVar();
        norm.setUpMeanAndStd(train);

        norm.normalize(train);
        norm.normalize(test);

        trainTarget = trainTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());
        testTarget = testTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());

        Classifier classifier = new SMO(c, eps, tol, max);
        classifier.fit(train, trainTarget);
        List<Double> predictLabels = classifier.predict(test);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predictLabels, testTarget);
        System.out.println(cm);
        return cm;
    }

    public static void spambase(double c, double eps, double tol, int max) throws Exception {
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

        trainTarget = trainTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());
        testTarget = testTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());

        Classifier clf = new SMO(c, eps, tol, max);

        List<Double> predicts = clf.fit(train, trainTarget).predict(train);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predicts, trainTarget);
        System.out.println("============== Train Performance==========\n" + cm);

        predicts = clf.predict(test);
        cm = Validation.confusionMatrix(predicts, testTarget);
        System.out.println("============== Test Performance==========\n" + cm);
    }

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
//        spambase(0.01, 0.001, 0.1, 50);
//        kfoldSpambase(0.01, 0.001, 0.1, 50);
        digital(12000);
        final long endTime = System.nanoTime();
        System.out.format("Total Run time: %f secs\n", 1.0 * (endTime - startTime) / 1e9);
    }
}
