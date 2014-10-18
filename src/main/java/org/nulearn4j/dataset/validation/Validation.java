package org.nulearn4j.dataset.validation;

import java.util.List;

/**
 * Created by jiachiliu on 10/18/14.
 */
public class Validation {

    public static double mse(List<Double> predicts, List<Double> target){
        double error = 0.0;
        for(int i=0; i<predicts.size(); i++){
            Double p = predicts.get(i);
            Double t = target.get(i);
            error += (p - t) * (p - t);
        }
        return error / predicts.size();
    }
}
