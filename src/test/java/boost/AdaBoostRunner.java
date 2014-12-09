package boost;

import net.nulearn4j.boost.AdaBoost;
import net.nulearn4j.dataset.loader.DatasetLoader;
import net.nulearn4j.dataset.matrix.Matrix;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 11/11/14.
 */
public class AdaBoostRunner {

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

        trainTarget = trainTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());
        testTarget = testTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());

        final long startTime = System.nanoTime();
        AdaBoost clf = new AdaBoost(300);
        clf.fit(train, trainTarget, test, testTarget);
        final long endTime = System.nanoTime();
        System.out.format("Total Run time: %f\n", 1.0 * (endTime - startTime) / 1e9);
    }

    public static void polluteSpam() throws Exception {
        System.out.format("Load Dataset...\n");
        Matrix<Double> train = DatasetLoader.loadPolluteSpamBaseTrain(" ");
        Matrix<Double> test = DatasetLoader.loadPolluteSpamBaseTest(" ");
        List<Double> trainTarget = DatasetLoader.loadLabel("data/spam_polluted/train_label.txt");
        List<Double> testTarget = DatasetLoader.loadLabel("data/spam_polluted/test_label.txt");

        System.out.format("Convert Dataset label to -1 and +1...\n");
        trainTarget = trainTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());
        testTarget = testTarget.stream().map((v) -> (v == 0) ? -1.0 : v).collect(Collectors.toList());

        System.out.format("Start boosting...\n");
        final long startTime = System.nanoTime();
        AdaBoost clf = new AdaBoost(300);
        clf.fit(train, trainTarget, test, testTarget);
        final long endTime = System.nanoTime();
        System.out.format("Total Run time: %f secs\n", 1.0 * (endTime - startTime) / 1e9);
    }

    public static void main(String[] args) throws Exception {
        spambase();
//        polluteSpam();
    }
}
