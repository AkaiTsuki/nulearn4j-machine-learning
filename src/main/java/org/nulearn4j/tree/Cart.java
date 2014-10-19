package org.nulearn4j.tree;

import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 10/18/14.
 */
public abstract class Cart {

    private Logger logger = LoggerFactory.getLogger(Cart.class);

    /**
     * The maximum level of tree (exclusive root).
     */
    protected int maxLevel = 4;

    /**
     * The minimum number of data in each node.
     */
    protected int minDataInTreeNode = 10;

    /**
     * A list that indicate the indices of each feature.
     */
    protected int[] features;

    /**
     * The index of label.
     */
    protected int label;

    /**
     * The root of the tree.
     */
    protected TreeNode root;

    public Cart(int[] features, int label) {
        this.features = features;
        this.label = label;
    }

    public Cart(int[] features, int label, int maxLevel, int minDataInTreeNode) {
        this.features = features;
        this.label = label;
        this.maxLevel = maxLevel;
        this.minDataInTreeNode = minDataInTreeNode;
    }

    /**
     * The matrix of train should contains label
     *
     * @param train train data matrix
     */
    public void fit(Matrix<Double> train) {
        logger.info("Start build Tree");
        this.root = buildTree(train, 0);
    }

    /**
     * Recursively build tree node based on given matrix
     *
     * @param train a matrix of train dataset
     * @param level the current level of the node
     * @return the tree node that represents the best filter feature
     */
    protected TreeNode buildTree(Matrix<Double> train, int level) {
        logger.debug("Build Tree on level {}, data size: {}", level, train.getRowCount());

        // If number of train data points is less than minimum requirement
        if (train.getRowCount() < minDataInTreeNode) return new TreeNode(level, label, majorityVote(train));
        // all data has same label
        if (isSameLabel(train)) return new TreeNode(level, label, majorityVote(train));
        // meet the max level
        if (level >= maxLevel) return new TreeNode(level, label, majorityVote(train));

        SplitCriteria splitCriteria = findBestSplitFeature(train);
        logger.debug("find best filter feature: {}", splitCriteria);
        if (splitCriteria == null) {
            return new TreeNode(level, label, majorityVote(train));
        }

        int f = splitCriteria.feature;
        double v = splitCriteria.value;
        Matrix<Double> left = train.filter((Row<Double> r) -> r.get(f) <= v);
        Matrix<Double> right = train.filter((Row<Double> r) -> r.get(f) > v);
        logger.debug("data filter into: [left: {}, right: {} ]", left.getRowCount(), right.getRowCount());

        TreeNode leftTree, rightTree;
        if (left.getRowCount() == 0) {
            leftTree = new TreeNode(level, label, majorityVote(train));
        } else {
            leftTree = buildTree(left, level + 1);
        }

        if (right.getRowCount() == 0) {
            rightTree = new TreeNode(level, label, majorityVote(train));
        } else {
            rightTree = buildTree(right, level + 1);
        }

        return new TreeNode(level, f, v, leftTree, rightTree);
    }

    /**
     * Given a dataset, find the best feature that can achieve the best score
     *
     * @param matrix a dataset
     * @return a split criteria saving information about the split value and score
     */
    protected SplitCriteria findBestSplitFeature(Matrix<Double> matrix) {
        double score = calculateScore(matrix);
        SplitCriteria sc = findBestSplitPoint(matrix, features[0], score);
        for (int i = 1; i < features.length; i++) {
            SplitCriteria sp = findBestSplitPoint(matrix, features[i], score);
            if (sc != null && sp != null && sp.compare(sc, sp) < 0) {
                sc = sp;
            }
        }
        return sc;
    }

    /**
     * Find the best split value with a feature
     *
     * @param matrix      a dataset
     * @param feature     the index of feature
     * @param parentScore the parent score that use to calculate the score reduction
     * @return a split criteria saving information about the split value and score
     */
    protected SplitCriteria findBestSplitPoint(Matrix<Double> matrix, int feature, double parentScore) {
        SplitCriteria sc = null;
        matrix.sortByFeature(feature);
        for (int i = 1; i < matrix.getRowCount(); i++) {
            if (matrix.get(i - 1, feature).equals(matrix.get(i, feature))) {
                continue;
            }
            double splitValue = (matrix.get(i - 1, feature) + matrix.get(i, feature)) / 2;
            Matrix<Double>[] splits = matrix.split(i);
            double score = calculateScore(splits[0], splits[1], parentScore);
            if (sc == null) {
                sc = new SplitCriteria(feature, splitValue, score);
            } else {
                SplitCriteria current = new SplitCriteria(feature, splitValue, score);
                if (current.compare(sc, current) < 0) {
                    sc = current;
                }
            }

        }
        return sc;
    }

    /**
     * Calculate the score of given dataset, the score can be entropy.
     *
     * @param matrix a dataset
     * @return the score of dataset
     */
    protected abstract double calculateScore(Matrix<Double> matrix);

    /**
     * Calculate the score in reduction based on splitted datasets
     *
     * @param left        a splits of dataset
     * @param right       a splits of dataset
     * @param parentScore the score on the dataset before splitting
     * @return the score in reduction
     */
    protected abstract double calculateScore(Matrix<Double> left, Matrix<Double> right, double parentScore);

    /**
     * Find the majority vote of the label for decision tree,
     * or the mean value of label for regression tree.
     *
     * @param matrix a dataset
     * @return the calculated label.
     */
    protected abstract double majorityVote(Matrix<Double> matrix);

    /**
     * test whether all labels of the matrix is same.
     *
     * @param matrix a dataset
     * @return true if there are all same label, otherwise false.
     */
    protected abstract boolean isSameLabel(Matrix<Double> matrix);

    public List<Double> predict(Matrix<Double> test) {
        return test.getRows()
                .stream()
                .map(this::predict)
                .collect(Collectors.toList());
    }

    public Double predict(Row<Double> row) {
        TreeNode node = this.root;
        while (node != null && node.feature != label) {
            if (row.get(node.feature) <= node.value) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.value;
    }

    public static void printTree(TreeNode root) {
        if (root != null) {
            System.out.println(root);
            printTree(root.getLeft());
            printTree(root.getRight());
        }
    }

    public TreeNode getRoot() {
        return root;
    }


    public static class SplitCriteria implements Comparator<SplitCriteria> {
        /**
         * feature index
         */
        public int feature;

        /**
         * the value of split point
         */
        public double value;

        /**
         * the calculate score
         */
        public Double score;

        public SplitCriteria() {
        }

        public SplitCriteria(int feature, double value, double score) {
            this.feature = feature;
            this.value = value;
            this.score = score;
        }

        @Override
        public String toString() {
            return "SplitCritera{" +
                    "feature=" + feature +
                    ", value=" + value +
                    ", score=" + score +
                    '}';
        }

        @Override
        public int compare(SplitCriteria o1, SplitCriteria o2) {
            return o1.score.compareTo(o2.score);
        }
    }

    public static class TreeNode {
        /**
         * level of the node in tree
         */
        private int level;

        /**
         * the feature the node split on
         */
        private int feature;

        /**
         * the split value
         */
        private double value;

        /**
         * left sub-tree
         */
        private TreeNode left;

        /**
         * right sub-tree
         */
        private TreeNode right;

        public TreeNode() {
        }

        public TreeNode(int level, int feature, double value) {
            this.level = level;
            this.feature = feature;
            this.value = value;
        }

        public TreeNode(int level, int feature, double value, TreeNode left, TreeNode right) {
            this.level = level;
            this.feature = feature;
            this.value = value;
            this.left = left;
            this.right = right;
        }

        public String toString() {
            String str = "|";
            for (int i = 0; i < level; i++) {
                str += "--";
            }
            str += "TreeNode{ " +
                    "level=" + level +
                    ", feature=" + feature +
                    ", value=" + value +
                    '}';
            return str;
        }

        public int getLevel() {
            return level;
        }

        public int getFeature() {
            return feature;
        }

        public double getValue() {
            return value;
        }

        public TreeNode getLeft() {
            return left;
        }

        public TreeNode getRight() {
            return right;
        }
    }
}
