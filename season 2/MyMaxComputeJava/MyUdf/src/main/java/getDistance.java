import com.aliyun.odps.udf.UDF;

import java.lang.Math;

public class getDistance extends UDF {
    // 0.1 ** (((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5 * 100000)
    public Double evaluate(String lon1, String lat1, String lon2, String lat2) {
        double lo1 = Double.parseDouble(lon1);
        double la1 = Double.parseDouble(lat1);
        double lo2 = Double.parseDouble(lon2);
        double la2 = Double.parseDouble(lat2);
        double distance = Math.sqrt(Math.pow((lo1 - lo2), 2) + Math.pow((la1 - la2), 2));
        return distance;
    }
}