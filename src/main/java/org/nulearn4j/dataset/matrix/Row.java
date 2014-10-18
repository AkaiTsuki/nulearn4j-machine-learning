package org.nulearn4j.dataset.matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class Row<T> {

    private List<T> data;

    public Row() {
        data = new ArrayList<>();
    }

    public Row(List<T> data) {
        this.data = data;
    }

    public Row(T[] data) {
        this.data = Stream.of(data).collect(Collectors.toList());
    }

    public T get(int i) {
        return data.get(i);
    }

    public int size() {
        return data.size();
    }

    public List<T> getData() {
        return data;
    }

    public void setData(List<T> data) {
        this.data = data;
    }

    public void add(T val){
        data.add(val);
    }

    public String toString(){
        return data.toString();
    }
}
