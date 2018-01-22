# 1 比赛

本赛题目标为在商场内精确的定位用户当前所在商铺。给出的信息包括wifi信号强度、GPS、基站定位、历史交易，来确定测试集交易发生的店铺。

我们队伍是`我去，咋回事`（[出门向右](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1095279130692) 、[东风西风读书屋](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1095279166608) 、[wakup](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1095279112728) 、[关山](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1059604433)）

详情和数据见[比赛官网](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.3.5afa4a94ovJNZV&raceId=231620)

# 2 数据与评价方式

提供了用户脱敏的2017-07-01 ~ 2017-08-31的交易详细数据（包括交易时wifi信号强度、GPS、基站定位）数据，预测用户2017-09月01~14日的交易发生的店铺。

评价方式：准确率=预测正确的shop个数/总样本数。

# 3 解决方案

主要有两部分，一是构造候选店铺集合，然后在候选集中做二分类预测。而构造候选如果没做好，后面预测就没有意义，所以构造候选使用了覆盖率的指标，在此基础上，最后使用准确率为最终指标，以便分步调优。

## 3.1 数据划分

|  集合  |           样本区间           |           特征区间           |
| :--: | :----------------------: | :----------------------: |
| 训练集  | [2017-08-25, 2017-08-31] | [2017-07-01, 2017-08-25) |
| 预测集  | [2017-09-01, 2017-09-14] | [2017-07-01, 2017-08-31] |

## 3.2 预处理

1. 有wifi强度为null的数据，直接删除掉，以免造成干扰。
2. 对于训练集出现次数小于3次的wifi过滤掉，一定程度可以减少bssid的数量。

## 3.3 构造候选

采用了多个构造候选集的方式，通过覆盖率来评估其效果，第一赛季覆盖率97%，第二赛季95%。主要有：

1. 连接过的wifi历史店铺。测试集连接wifi的记录中，取出bssid，与特征区间连接wifi的记录中，找到相同bssid记录的计数前n店铺。

2. TF-IDF选取前3样本。

   $$TF-IDF = TF(词频) * IDF(逆文档频率)$$

   此项目中，同一记录根据wifi信号强度排序获得排序值，并做weight=f(x)=exp((0 - i) * 0.6)映射。

   1. 对于特征区间，定义shop_tfidf =shop-bssid分组求weight和/(shop分组求weight和 * bssid分组求weight和)，
   2. 对于样本区间，对此商场的每个店铺，计算其和此样本所有bssid的tfidf值（通过1中join）并求和作为此shop的tfidf。然后取tfidf值排名前n。

3. 最强信号的采样：

   特征区间店铺交易的最强wifi的bssid做计数，然后在样本区间最强的bssid关联之前的店铺计数，取前n个。

4. 用户在此商场去过的商店次数最多的n个。

5. 根据记录发生的经纬度与店铺交易经纬度最近的n个。（用特征区间商铺交易发生的经纬度代替其本身经纬度）。

6. 根据记录发生的经纬度与店铺本身经纬度最近的n个。

7. wifi信号的cos相似度最近似的n个。对于wifi信号信息，把它看做高维向量，就可以根据向量的cos相似度来计算wifi信号相似度。

距离计算使用公式
```
0.1 ** (((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2) ** 0.5 * 100000)
```

我们n取3或者4左右。

## 3.4 二分类预测

通过上一步构建候选集，这一步主要完成是否此店铺的问题，即二分类。

### 特征

#### wifi特征

- 连接wifi与此店铺交易时连接wifi的次数。
- 店铺与此记录的tfidf值（见构造候选）
- 样本区间此记录最强信号与店铺历史交易最强信号相同的计数。
- 样本区间此记录wifi信号强度与店铺历史wifi余弦相似度。
- 是否连接wifi。
- 样本区间记录的wifi信号强度排名和店铺wifi信号强度排名，作为两个向量，计算L1,L2距离。
- 样本区间wifi与店铺历史wifi中有同样的bssid的个数。wifi_count_sum
- wifi_count_sum/店铺的历史wifi计数

#### 距离特征

- 样本区间记录发生的经纬度与店铺交易平均经纬度的距离
- 样本区间记录发生的经纬度与店铺交易经纬度的函数映射求和。
- 距离店铺经纬度距离。

#### 用户、商店特征

- 商店交易次数。
- 商店交易次数/商场次数 = 在商场占比。
- 分小时，商店在记录所在小时段的交易量（均值最大最小），占比。
- 分周末，商店在周末和非周末的交易量特征。
- 用户去过此商店次数
- 用户去过此商店次数/用户次数 = 在用户占比。
- 用户去过此商店次数/用户此商场次数 = 在用户此商场占比。
- 用户在此price区间的消费次数。
- 用户平均的price-此记录price。

其他一些特征可以参考代码，在此不赘述。

### 算法模型

初赛使用了XGBoost和lightGBM，lightGBM效果优于xgboost，复赛使用XGBoost和GBDT（XGBoost>GBDT）而且GBDT巨耗费能量，后期也是优于计算量的限制放弃了blending的融合方法。

### 模型融合

前期使用了blending的融合方法，将训练集分为两部分，然后第一部分用于训练基模型，及基模型的概率值作为第二部分的特征，来训练第二部分，然后预测测试集。微笑提升，但是特别消耗计算量。

后期使用多个模型概率值加权融合，微小提升。



# 感想

此次队友给力([出门向右](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1095279130692) 、[东风西风读书屋](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1095279166608) 、[wakup](https://tianchi.aliyun.com/science/scientistDetail.htm?userId=1095279112728) )，主要负责线下，我负责线上赛，复赛我们没有使用多分类构建特征（主要是考虑计算资源不够），是一个大的失误，据说提升2个点左右，有点遗憾。此次计算资源也比较紧张，造成许多想法没能实现。

GitHub代码：[https://github.com/InsaneLife/Positioning-shops](https://github.com/InsaneLife/Positioning-shops)

