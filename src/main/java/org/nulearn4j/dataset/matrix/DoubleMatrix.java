package org.nulearn4j.dataset.matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class DoubleMatrix implements Matrix<Double> {

    private List<Row<Double>> matrix;

    public DoubleMatrix(){
        matrix = new ArrayList<>();
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
        return matrix.stream().map(row -> row.get(i)).collect(Collectors.toList());
    }

    @Override
    public Row<Double> getRow(int i) {
        return matrix.get(i);
    }

    public List<Row<Double>> getMatrix() {
        return matrix;
    }

    @Override
    public int getColumnCount() {
        return matrix.get(0).size();
    }

    @Override
    public int getRowCount() {
        return matrix.size();
    }

    @Override
    public Double get(int row, int col) {
        return matrix.get(row).get(col);
    }

    @Override
    public void add(Row<Double> row) {
        matrix.add(row);
    }

    public String toString() {
        return matrix.stream().map(Object::toString).collect(Collectors.joining("\n"));
    }
}
