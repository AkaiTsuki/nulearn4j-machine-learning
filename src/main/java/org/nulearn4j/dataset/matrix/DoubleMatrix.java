package org.nulearn4j.dataset.matrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class DoubleMatrix implements Matrix<Double> {

    private List<Row<Double>> rows;

    public DoubleMatrix() {
        rows = new ArrayList<>();
    }

    public DoubleMatrix(List<Row<Double>> rows) {
        this.rows = rows;
    }

    @Override
    public int[] getDimension() {
        int[] dimension = new int[2];

        dimension[0] = getRowCount();
        dimension[1] = getColumnCount();

        return dimension;
    }

    @Override
    public List<Double> getColumn(int i) {
        return rows.stream().map(row -> row.get(i)).collect(Collectors.toList());
    }

    @Override
    public Row<Double> getRow(int i) {
        return rows.get(i);
    }

    public List<Row<Double>> getRows() {
        return rows;
    }

    @Override
    public int getColumnCount() {
        return rows.get(0).size();
    }

    @Override
    public int getRowCount() {
        return rows.size();
    }

    @Override
    public Double get(int row, int col) {
        return rows.get(row).get(col);
    }

    @Override
    public void set(int row, int col, Double val) {
        Row<Double> r = getRow(row);
        r.set(col, val);
    }

    @Override
    public void add(Row<Double> row) {
        rows.add(row);
    }

    @Override
    public void addAll(List<Row<Double>> rows) {
        rows.addAll(rows);
    }

    @Override
    public Matrix<Double> removeColumn(int col) {
        Matrix<Double> newMatrix = new DoubleMatrix();
        for (Row<Double> row : rows) {
            Row<Double> newRow = new Row<>();
            for (int i = 0; i < row.size(); i++) {
                if (i == col) continue;
                newRow.add(row.get(i));
            }
            newMatrix.add(newRow);
        }
        return newMatrix;
    }

    @Override
    public Matrix<Double> addColumn(int col, Double val) {
        Matrix<Double> newMatrix = new DoubleMatrix();
        for (Row<Double> row : rows) {
            Row<Double> newRow = new Row<>();
            boolean added = false;
            for (int i = 0; i < row.size(); i++) {
                if (i == col && !added) {
                    newRow.add(val);
                    i--;
                    added = true;
                } else {
                    newRow.add(row.get(i));
                }
            }
            newMatrix.add(newRow);
        }
        return newMatrix;
    }

    @Override
    public void shuffle() {
        Collections.shuffle(rows);
    }

    @Override
    public Matrix<Double>[] split(int rowIndex) {
        Matrix<Double>[] splits = new DoubleMatrix[2];

        splits[0] = new DoubleMatrix(rows.subList(0, rowIndex));
        splits[1] = new DoubleMatrix(rows.subList(rowIndex, rows.size()));

        return splits;
    }

    @Override
    public Matrix<Double> filter(Predicate<Row<Double>> predicate) {
        List<Row<Double>> filtered = rows.stream().filter(predicate).collect(Collectors.toList());
        return new DoubleMatrix(filtered);
    }

    @Override
    public Matrix<Double> kFoldTrain(int folds, int n) {
        int numOfRowsPerFold = getRowCount() / folds;
        int offsetStart = n * numOfRowsPerFold;
        int offsetEnd = n * numOfRowsPerFold + numOfRowsPerFold;
        Matrix<Double> m = new DoubleMatrix();

        List<Row<Double>> left = rows.subList(0, offsetStart);
        List<Row<Double>> right = rows.subList(offsetEnd, getRowCount());

        for (Row<Double> r : left) {
            m.add(new Row<>(r));
        }

        for (Row<Double> r : right) {
            m.add(new Row<>(r));
        }

        return m;
    }

    @Override
    public Matrix<Double> kFoldTest(int folds, int n) {
        int numOfRowsPerFold = getRowCount() / folds;
        int offsetStart = n * numOfRowsPerFold;
        int offsetEnd = n * numOfRowsPerFold + numOfRowsPerFold;
        List<Row<Double>> selected = rows.subList(offsetStart, offsetEnd);

        Matrix<Double> m = new DoubleMatrix();
        for (Row<Double> r : selected) {
            m.add(new Row<>(r));
        }
        return m;
    }

    @Override
    public void sortByFeature(int feature) {
        Collections.sort(rows, (r1, r2) -> r1.get(feature).compareTo(r2.get(feature)));
    }

    @Override
    public Double[][] to2DArray() {
        Double[][] array = new Double[getRowCount()][getColumnCount()];

        for (int i = 0; i < getRowCount(); i++) {
            for (int j = 0; j < getColumnCount(); j++) {
                array[i][j] = get(i, j);
            }
        }

        return array;
    }

    @Override
    public double[][] toPrimitive() {
        double[][] array = new double[getRowCount()][getColumnCount()];

        for (int i = 0; i < getRowCount(); i++) {
            for (int j = 0; j < getColumnCount(); j++) {
                array[i][j] = get(i, j);
            }
        }

        return array;
    }

    public String toString() {
        return rows.stream().map(Object::toString).collect(Collectors.joining("\n"));
    }
}
