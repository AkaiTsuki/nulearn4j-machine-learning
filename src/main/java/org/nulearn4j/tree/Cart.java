package org.nulearn4j.tree;

import org.nulearn4j.dataset.matrix.Matrix;
import org.nulearn4j.dataset.matrix.Row;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 10/18/14.
 */
public abstract class Cart {

    protected int maxLevel = 4;
    protected int minDataInTreeNode = 10;
    protected int[] features;
    protected int label;
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
        System.out.println("Start build Tree");
        this.root = buildTree(train, 0);
    }

    protected TreeNode buildTree(Matrix<Double> train, int level) {
        System.out.println("Build Tree on level " + level + " data size: "+ train.getRowCount());
        if (train.getRowCount() < minDataInTreeNode) return new TreeNode(level, label, majorityVote(train));
        if (isSameLabel(train)) return new TreeNode(level, label, majorityVote(train));
        if (level >= maxLevel) return new TreeNode(level, label, majorityVote(train));

        SplitCriteria splitCriteria = findBestSplitFeature(train);
        System.out.println("find best split feature: " + splitCriteria);
        if (splitCriteria == null) {
            return new TreeNode(level, label, majorityVote(train));
        }

        int f = splitCriteria.feature;
        double v = splitCriteria.value;
        Matrix<Double> left = train.split((Row<Double> r) -> r.get(f) <= v);
        Matrix<Double> right = train.split((Row<Double> r) -> r.get(f) > v);
        System.out.println("data split into: [left: " + left.getRowCount()+", right: "+right.getRowCount()+" ]");

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

    protected SplitCriteria findBestSplitFeature(Matrix<Double> matrix) {
        double score = calculateScore(matrix);
        SplitCriteria sc = findBestSplitPoint(matrix, features[0], score);
        for (int i = 1; i < features.length; i++) {
            SplitCriteria sp = findBestSplitPoint(matrix, features[i], score);
            if (sp != null && sp.compare(sc, sp) < 0) {
                sc = sp;
            }
        }
        return sc;
    }

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

    protected abstract double calculateScore(Matrix<Double> matrix);

    protected abstract double calculateScore(Matrix<Double> left, Matrix<Double> right, double parentScore);

    protected abstract double majorityVote(Matrix<Double> matrix);

    protected abstract boolean isSameLabel(Matrix<Double> matrix);

    public List<Double> predict(Matrix<Double> test) {
        return test.getRows()
                .stream()
                .map(this::predict)
                .collect(Collectors.toList());
    }

    public Double predict(Row<Double> row){
        TreeNode node = this.root;
        while(node!=null && node.feature!=label){
            if(row.get(node.feature) <= node.value){
                node = node.left;
            }else{
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

    public TreeNode getRoot(){
        return root;
    }

    public static class SplitCriteria implements Comparator<SplitCriteria> {
        public int feature;
        public double value;
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
        private int level;
        private int feature;
        private double value;
        private TreeNode left;
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
