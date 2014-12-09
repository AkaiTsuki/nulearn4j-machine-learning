package net.nulearn4j.dataset.matrix;

import java.util.List;
import java.util.function.Predicate;

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
     * @param row a row
     */
    void add(Row<T> row);

    /**
     * Add all given rows into end of matrix
     *
     * @param rows a list of rows
     */
    void addAll(List<Row<T>> rows);

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

    /**
     * Split the matrix based on given predicate
     *
     * @param func a predicate function that use as filter
     * @return a new matrix that pass the predicate.
     */
    Matrix<T> filter(Predicate<Row<T>> func);

    /**
     * @param folds total number of folds
     * @param n     the current fold that will choose as test
     * @return a new matrix that contains rows not in this fold
     */
    Matrix<T> kFoldTrain(int folds, int n);

    /**
     * @param folds total number of folds
     * @param n     the current fold that will choose as test
     * @return a new matrix that contains rows in this fold
     */
    Matrix<T> kFoldTest(int folds, int n);

    /**
     * Sort all the rows by given feature
     *
     * @param feature index of a feature
     */
    void sortByFeature(int feature);

    T[][] to2DArray();

    double[][] toPrimitive();
}
