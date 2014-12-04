package org.nulearn4j.neighbor;

/**
 * Created by jiachiliu on 12/4/14.
 */
public class KernelFactory {
    public static Kernel getInstance(String kernel){
        switch (kernel) {
            case Kernel.EUCLIDIAN:
                return new EuclidianDistanceKernel();
            case Kernel.COSINE:
                return new CosineKernel();
            case Kernel.GAUSSIAN:
                return new GaussianKernel();
            case Kernel.POLY:
                return new PolyKernel();
            default:
                return new EuclidianDistanceKernel();
        }
    }
}
