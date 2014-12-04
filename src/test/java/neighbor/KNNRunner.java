package neighbor;

import org.nulearn4j.dataset.loader.DatasetLoader;
import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.preprocessing.normalization.Normalization;
import org.nulearn4j.dataset.preprocessing.normalization.ZeroMeanUnitVar;
import org.nulearn4j.neighbor.ClassificationKNN;
import org.nulearn4j.neighbor.KNN;
import org.nulearn4j.neighbor.Kernel;
import org.nulearn4j.validation.Validation;

import java.util.Arrays;
import java.util.List;

/**
 * Created by jiachiliu on 12/3/14.
 */
public class KNNRunner {

    public static void spambase(int k, String kernel) throws Exception {
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

        KNN clf = new ClassificationKNN(k, kernel);
        System.out.println(clf);

        List<Double> predicts = clf.predict(train, trainTarget, test);
        Validation.ConfusionMatrix cm = Validation.confusionMatrix(predicts, testTarget);
        System.out.println("============== Test Performance==========\n" + cm);
    }

    public static void digital(int size, int k, String kernel) throws Exception {
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

        KNN clf = new ClassificationKNN(k, kernel);
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

        spambase(7, Kernel.EUCLIDIAN);
//        digital(12000, 7, Kernel.POLY);

        final long endTime = System.nanoTime();
        System.out.format("Total Run time: %f secs\n", 1.0 * (endTime - startTime) / 1e9);
    }
}
