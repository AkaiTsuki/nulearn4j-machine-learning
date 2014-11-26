package svm;

import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.preprocessing.normalization.Normalization;
import org.nulearn4j.dataset.preprocessing.normalization.ZeroMeanUnitVar;
import org.nulearn4j.svm.SMO;
import org.nulearn4j.validation.Validation;

import java.util.List;
import java.util.stream.Collectors;


public class SMORunner {

    public static void kfoldSpambase() throws Exception {
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
            Validation.ConfusionMatrix cm = run(test, train);
            accuracy += cm.accuracy();
            error += cm.error();
        }

        System.out.println("Average Accuracy: " + accuracy / k + " Average error: " + error / k);
    }

    private static Validation.ConfusionMatrix run(Matrix<Double> test, Matrix<Double> train) throws Exception {
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

        SMO classifier = new SMO();
        classifier.fit(train, trainTarget);
        List<Double> predictLabels = classifier.predictToLabel(test);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predictLabels, testTarget);
        System.out.println(cm);
        return cm;
    }

    public static void spambase() throws Exception {
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

        SMO clf = new SMO();
        clf.fit(train, trainTarget);

        List<Double> predicts = clf.predictToLabel(train);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predicts, trainTarget);
        System.out.println("============== Train Performance==========\n" + cm);

        predicts = clf.predictToLabel(test);
        cm = Validation.confusionMatrix(predicts, testTarget);
        System.out.println("============== Test Performance==========\n" + cm);
    }

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
//        spambase();
        kfoldSpambase();
        final long endTime = System.nanoTime();
        System.out.format("Total Run time: %f secs\n", 1.0 * (endTime - startTime) / 1e9);
    }
}
