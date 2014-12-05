package org.nulearn4j.neighbor;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by jiachiliu on 12/4/14.
 */
public class Configuration {
    private Map<String, String> params = new HashMap<>();

    public void set(String name, String value) {
        params.put(name, value);
    }

    public String get(String name) {
        return params.get(name);
    }

    public void setDouble(String name, Double value) {
        params.put(name, value.toString());
    }

    public double getDouble(String name) {
        return new Double(params.get(name));
    }
}
