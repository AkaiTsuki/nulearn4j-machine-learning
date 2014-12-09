package net.nulearn4j.neighbor;

/**
 * Created by jiachiliu on 12/4/14.
 */
public class KernelFactory {
    public static Kernel getInstance(Configuration config) {
        switch (config.get("kernel")) {
            case Kernel.EUCLIDIAN:
                return new EuclidianDistanceKernel();
            case Kernel.COSINE:
                return new CosineKernel();
            case Kernel.GAUSSIAN:
                if (config.get("gamma") != null) {
                    return new GaussianKernel(config.getDouble("gamma"));
                } else {
                    return new GaussianKernel();
                }
            case Kernel.POLY:
                if (config.get("degree") != null && config.get("C") != null) {
                    return new PolyKernel(config.getDouble("degree"), config.getDouble("C"));
                }
                return new PolyKernel();
            default:
                return new EuclidianDistanceKernel();
        }
    }
}
