package com.aliyun.odps.examples.udf.test;

import com.aliyun.odps.Odps;
import com.aliyun.odps.account.Account;
import com.aliyun.odps.account.AliyunAccount;

public class TestUtil {
  private final static String accessId = "accessId";
  private final static String accessKey = "accessKey";
  private final static String endpoint = "endpoint";
  private final static String defaultProject = "example_project";

  static Odps odps;
  static {
    Account account = new AliyunAccount(accessId, accessKey);
    odps = new Odps(account);
    odps.setEndpoint(endpoint);
    odps.setDefaultProject(defaultProject);
  }

  public static String join(Object[] obj) {
    if (obj == null) {
      return null;
    }
    StringBuffer sb = new StringBuffer();
    for (int i = 0; i < obj.length; i++) {
      if (sb.length() > 0) {
        sb.append(",");
      }
      sb.append(obj[i]);
    }
    return sb.toString();
  }

  public static Odps getOdps() {
    return odps;
  }
}
