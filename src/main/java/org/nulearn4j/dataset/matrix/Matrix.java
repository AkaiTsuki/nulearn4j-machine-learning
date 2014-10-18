package org.nulearn4j.dataset.matrix;

import java.util.List;

/**
 * Created by jiachiliu on 10/17/14.
 */
public interface Matrix<T> {
    /**
     * @return the dimension of matrix in a array
     * the first element is the number of rows
     * the second element is the number of columns
     */
    int[] getDimension();

    /**
     * Return all values in given column
     *
     * @param i the index of column
     * @return a list of values in given column
     */
    List<T> getColumn(int i);

    /**
     * Return a row on the given row index
     *
     * @param i row index
     * @return Row
     */
    Row<T> getRow(int i);

    /**
     * get all the rows in matrix
     *
     * @return a list of Rows
     */
    List<Row<T>> getRows();

    /**
     * @return number of columns in matrix
     */
    int getColumnCount();

    /**
     * @return number of rows in matrix
     */
    int getRowCount();

    /**
     * @param row row index
     * @param col column index
     * @return the value under the given row and column
     */
    T get(int row, int col);

    /**
     * Set the given position with given value
     *
     * @param row row index
     * @param col column index
     */
    void set(int row, int col, T val);

    /**
     * Add a row
     *
     * @param row
     */
    void add(Row<T> row);

    /**
     * Remove the given column
     *
     * @param col column index
     * @return a new matrix
     */
    Matrix<T> removeColumn(int col);

    /**
     * Add a new column at given column with the value
     *
     * @param col column index
     * @param val the value for each element in new column
     * @return new matrix that has the new column
     */
    Matrix<T> addColumn(int col, T val);

    /**
     * shuffle all the rows in matrix
     */
    void shuffle();

    /**
     * Split the matrix into to smaller matrix
     *
     * @param rowIndex the row index
     * @return two matrices that one is [, rowIndex) and another is [rowIndex, )
     */
    Matrix<T>[] split(int rowIndex);
}
