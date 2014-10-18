package org.nulearn4j.dataset.matrix;

import java.util.List;

/**
 * Created by jiachiliu on 10/17/14.
 */
public interface Matrix<T> {
    /**
     *
     * @return the dimension of matrix in a array
     * the first element is the number of rows
     * the second element is the number of columns
     */
    int[] getDimension();

    /**
     * Return all values in given column
     * @param i the index of column
     * @return a list of values in given column
     */
    List<T> getColumn(int i);

    /**
     * Return a row on the given row index
     * @param i row index
     * @return Row
     */
    Row<T> getRow(int i);

    /**
     * get the underlying matrix
     * @return a list of Rows
     */
    List<Row<T>> getMatrix();

    /**
     *
     * @return number of columns in matrix
     */
    int getColumnCount();

    /**
     *
     * @return number of rows in matrix
     */
    int getRowCount();

    /**
     *
     * @param row row index
     * @param col column index
     * @return the value under the given row and column
     */
    T get(int row, int col);

    /**
     * Add a row
     * @param row
     */
    void add(Row<T> row);
}
