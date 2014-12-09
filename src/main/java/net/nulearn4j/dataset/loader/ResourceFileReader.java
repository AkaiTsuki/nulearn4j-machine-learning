package net.nulearn4j.dataset.loader;

import java.io.File;
import java.net.URL;

/**
 * Created by jiachiliu on 10/17/14.
 */
public class ResourceFileReader {
    public File read(String path) throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        URL url = classLoader.getResource(path);
        if (url == null) {
            throw new Exception("get resource from path: " + path + " returns null.");
        }
        return new File(url.getFile());
    }
}
