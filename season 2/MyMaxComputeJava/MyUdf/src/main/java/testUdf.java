import com.aliyun.odps.udf.UDF;

public final class testUdf extends UDF {
    public String evaluate(String s) {
        if (s == null) { return null; }
        return "hello world:" + s;
    }
}