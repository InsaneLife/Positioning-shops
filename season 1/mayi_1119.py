import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
import math
# from tqdm import tqdm
# import multiprocessing
# from collections import Counter
from collections import defaultdict

# from joblib import Parallel, delayed

cache_path = '/home/csu/myjf/mayi_cache/'
data_path = '/home/csu/myjf/data/test/'
test_path = data_path + 'evaluation_public_b.csv'  # B榜数据
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'  # 已经预处理，增加与测试集不冲突的row_id

flag = True


# 线下测评
def acc(data, name='shop_id'):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path, 'r'))
    except:
        print('没有发现真实数据，无法测评')
    return sum(data['row_id'].map(true) == data[name]).astype(float) / data.shape[0]


# 线下测评
def get_label(data):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path, 'r'))
    except:
        print('没有发现真实数据，无法测评')
    data['label'] = (data['shop_id'] == data['row_id'].map(true)).astype(int)
    return data


# 分组标准化
def grp_standard(data, key, names):
    for name in names:
        mean_std = data.groupby(key, as_index=False)[name].agg({'mean': 'mean',
                                                                'std': 'std'})
        data = data.merge(mean_std, on=key, how='left')
        data[name] = ((data[name] - data['mean']) / data['std']).fillna(0)
        data[name] = data[name].replace(-np.inf, 0)
        data.drop(['mean', 'std'], axis=1, inplace=True)
    return data


# 分组归一化
def grp_normalize(data, key, names, start=0):
    for name in names:
        max_min = data.groupby(key, as_index=False)[name].agg({'max': 'max',
                                                               'min': 'min'})
        data = data.merge(max_min, on=key, how='left')
        data[name] = (data[name] - data['min']) / (data['max'] - data['min'])
        data[name] = data[name].replace(-np.inf, start)
        data.drop(['max', 'min'], axis=1, inplace=True)
    return data


# 分组排序
def grp_rank(data, key, names, ascending=True):
    for name in names:
        data.sort_values([key, name], inplace=True, ascending=ascending)
        data['rank'] = range(data.shape[0])
        min_rank = data.groupby(key, as_index=False)['rank'].agg({'min_rank': 'min'})
        data = pd.merge(data, min_rank, on=key, how='left')
        data['rank'] = data['rank'] - data['min_rank']
        data[names] = data['rank']
        data.drop(['rank', 'min_rank'], axis=1, inplace=True)
    return data


# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result


# 分组rank
def group_rank(data, key, values, ascending=True):
    data.sort_values([key, values], inplace=True, ascending=ascending)  # 按key和value做ascending排序，默认升序
    data['rank'] = range(data.shape[0])  # 添加递增序列
    min_rank = data.groupby(key, as_index=False)['rank'].agg({'min_rank': 'min'})  # 按照key进行分组
    data = pd.merge(data, min_rank, on=key, how='left')
    data['rank'] = data['rank'] - data['min_rank']  # 按照key分类对values做rank
    del data['min_rank']
    return data


# 商店对应的连接wiif
def get_connect_wifi(wifi_infos):
    if wifi_infos != '':
        for wifi_info in wifi_infos.split(';'):
            bssid, signal, flag = wifi_info.split('|')
            if flag == 'true':
                return bssid  # 返回连接的wifi_bssid
    return np.nan


# 商店对应的连接wiif
def get_most_wifi(wifi_infos):
    if wifi_infos != '':
        bssid = sorted([wifi.split('|') for wifi in wifi_infos.split(';')], key=lambda x: int(x[1]), reverse=True)[0][
            0]  # 按强度进行排序，返回最强bssid
        return bssid
    return np.nan


# 排名对应的权重
def rank_weight(i):
    return np.exp((0 - i) * 0.6)


# 获取行对应的wifi信息
def get_row_wifi_infos_dict(train, test, data_key):
    result_path = cache_path + 'shop_cwifi_{}.pickle'.format(data_key)
    if os.path.exists(result_path) & flag:
        row_wifi_infos_dict = pickle.load(open(result_path, '+rb'))
    else:
        row_wifi_infos_dict = {}
        for row in train.itertuples():
            for wifi_info in row.wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                row_wifi_infos_dict[bssid] = signal
        for row in test.itertuples():
            for wifi_info in row.wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                row_wifi_infos_dict[bssid] = signal
        pickle.dump(row_wifi_infos_dict, open(result_path, '+wb'))
    return row_wifi_infos_dict


# wifi连接过的商店的个数
def get_shop_cwifi_count(data, data_key):
    result_path = cache_path + 'shop_cwifi_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp['bssid'] = data_temp['wifi_infos'].apply(get_connect_wifi)  # 返回连接的wifibssid，否则返回nan
        result = data_temp.groupby(['bssid', 'shop_id'], as_index=False)['shop_id'].agg(
            {'shop_cwifi_count': 'count'})  # 店铺连接的wifi数量，其实就是对bssid,shop_id进行分组统计
        result.sort_values(['bssid', 'shop_cwifi_count'], ascending=False, inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 获取商店-wifi对应的次数
def get_shop_wifi_tfidf(data, data_key):
    result_path = cache_path + 'shop_wifi_tfidf_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, signal])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'Flag'])  # 此处Flag是否是笔误
        shop_wifi = group_rank(shop_wifi, 'row_id', 'Flag', ascending=True)  # row_id进行分组，对信号强度进行升序排序，值越小强度越大
        shop_wifi['rank'] = shop_wifi['rank'].apply(rank_weight)  # 对不同的排序赋权值   np.exp((0 - i) * 0.6)
        shop_wifi.rename(columns={'rank': 'weight'}, inplace=True)
        shop_wifi_tfidf = shop_wifi.groupby(['shop_id', 'bssid'], as_index=False)['weight'].agg(
            {'shop_wifi_tfidf': 'sum'})
        shop_count = shop_wifi.groupby(['shop_id'], as_index=False)['weight'].agg({'shop_count': 'sum'})
        shop_wifi_tfidf = shop_wifi_tfidf.merge(shop_count, on='shop_id', how='left')
        shop_wifi_tfidf['tfidf'] = shop_wifi_tfidf['shop_wifi_tfidf'] / shop_wifi_tfidf['shop_count']
        wifi_count = shop_wifi_tfidf.groupby(['bssid'], as_index=False)['tfidf'].agg({'wifi_count': 'sum'})
        shop_wifi_tfidf = shop_wifi_tfidf.merge(wifi_count, on='bssid', how='left')
        shop_wifi_tfidf['tfidf'] = shop_wifi_tfidf['tfidf'] / shop_wifi_tfidf['wifi_count']
        result = shop_wifi_tfidf
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 最强wifi对应过的商店
def get_shop_mwifi_count(data, data_key):
    result_path = cache_path + 'shop_mwifi_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp['bssid'] = data_temp['wifi_infos'].apply(get_most_wifi)
        result = data_temp.groupby(['bssid', 'shop_id'], as_index=False)['shop_id'].agg({'shop_mwifi_count': 'count'})
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 清洗无用wifi #清洗的阈值是否还可以优化
def clear(data, candidate, data_key):  # 训练集，测试集，hash码
    data_path = cache_path + 'data_clear_{}.hdf'.format(data_key)  # 存放hash码的地址
    candidate_path = cache_path + 'candidate_clear_{}.hdf'.format(data_key)
    if os.path.exists(data_path) & os.path.exists(candidate_path) & flag:  # 、如果已经存在，读取
        data = pd.read_hdf(data_path, 'w')
        candidate = pd.read_hdf(candidate_path, 'w')
    else:
        wifi_count_signal_dict = defaultdict(lambda: 0)  # 记录训练集中每个wifi出现的次数
        for wifi_infos in data['wifi_infos'].values:  # 遍历每一个wifi_infos
            for wifi_info in wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                wifi_count_signal_dict[bssid] += 1  # 出现次数统计

        def f_1(wifi_infos):  # wifi次数大于1的连接起来
            return ';'.join(
                [wifi_info for wifi_info in wifi_infos.split(';') if
                 wifi_count_signal_dict[wifi_info.split('|')[0]] > 1])

        def f_2(wifi_infos):  # wifi次数大于0的连接起来
            return ';'.join(
                [wifi_info for wifi_info in wifi_infos.split(';') if
                 wifi_count_signal_dict[wifi_info.split('|')[0]] > 0])

        data['wifi_infos'] = data['wifi_infos'].apply(f_1)  # 保留出现次数>1的wifi_info
        candidate['wifi_infos'] = candidate['wifi_infos'].apply(f_2)  # 测试集中保留在训练集出现次数>0的结果
        data.to_hdf(data_path, 'w', complib='blosc', complevel=5)  # 保存数据
        candidate.to_hdf(candidate_path, 'w', complib='blosc', complevel=5)  # 保存数据
    return (data, candidate)


###################################################
# ................... 选取备选样本 ...................
###################################################
# 用连接的wifi选取样本
def get_cwifi_sample(data, candidate, data_key):  # 找出dataa里连接wifi的排序shop_id，candidate里相同连接的bssid也可以选用这些shop_id作为候选集
    result_path = cache_path + 'cwifi_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()  # 复制一份数据
        candidate_temp['bssid'] = candidate_temp['wifi_infos'].apply(get_connect_wifi)  # 没有连接wifi的都被提速为nan
        shop_cwifi_count = get_shop_cwifi_count(data, data_key)
        shop_cwifi_count = group_rank(shop_cwifi_count, 'bssid', 'shop_cwifi_count',
                                      ascending=False)  # 按bssid分组，进行降序排序count大，排序靠前
        shop_cwifi_count = shop_cwifi_count[shop_cwifi_count['rank'] < 3]  # 取排序前三
        # 进行内连接，只有candidate_temp和shop_cwifi_count都存在的bssid才会进行连接，排序前三的shop_id都会合并
        result = candidate_temp.merge(shop_cwifi_count, on='bssid', how='inner')
        result = result[['row_id', 'shop_id']]  # 只返回row_id和shop_id
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result  # 通过bssid连接的店铺排序进行样本选择，即只选取了wifi有连接wifi的row_id,并选取排序前三的shop_id作为候选样本


# 用简单tfidf选取前3样本
def get_tfidf_sample(data, candidate, data_key):
    result_path = cache_path + 'tfidf_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_tfidf = get_shop_wifi_tfidf(data, data_key)  # 获取tfidf
        shop_wifi_tfidf_dict = {}  # shop_id:{bssid:tfidf}
        for shop_id, grp in shop_wifi_tfidf.groupby('shop_id'):  # 对tfidf信息进行shop分组
            wifi_idf = {}
            for tuple in grp.itertuples():
                wifi_idf[tuple.bssid] = tuple.tfidf  # bssid:tfidf 字典
                shop_wifi_tfidf_dict[shop_id] = wifi_idf
        shop = pd.read_csv(shop_path)
        mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()
        result = []
        for row in candidate.itertuples():  # 处理测试集
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                wifi_infos = sorted([wifi.split('|') for wifi in wifi_infos.split(';')], key=lambda x: int(x[1]),
                                    reverse=True)  # 根据强度降序排序
                shops = mall_shop_dict[row.mall_id]  # mall_id的shop_id列表[]
                for shop_id in shops:  # shops
                    shop_tfidf = 0
                    for i, (bssid, signal, Flag) in enumerate(wifi_infos):  # wifi_infos进行排序，遍历列表
                        try:
                            idf = shop_wifi_tfidf_dict[shop_id][bssid] * rank_weight(
                                i)  # shop_wifi_tfidf_dict保存不同店铺不同bssid的tfidf值
                            shop_tfidf += idf  # 把同一店铺不同bssid的tfidf累加起来，作为shop_tfidf
                        except:
                            pass
                    if shop_tfidf > 0:  # 如果店铺的tfidf>0,候选
                        result.append([row.row_id, shop_id, shop_tfidf])
        result = pd.DataFrame(result, columns=['row_id', 'shop_id', 'shop_tfidf'])
        result = group_rank(result, 'row_id', 'shop_tfidf', ascending=False)  # 对shop_tfidf进行排序
        result = result[result['rank'] < 3][['row_id', 'shop_id']]  # 保留前三
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用最强信号选取前二样本
def get_mwifi_sample(data, candidate, data_key):
    result_path = cache_path + 'mwifi_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp['bssid'] = candidate_temp['wifi_infos'].apply(get_most_wifi)  # 获取最强bssid
        shop_mwifi_count = get_shop_mwifi_count(data, data_key)  # 最强wifi对应的店铺
        shop_mwifi_count = group_rank(shop_mwifi_count, 'bssid', 'shop_mwifi_count',
                                      ascending=False)  # 最强bssid分组，在店铺出现的次数进行排序，
        shop_mwifi_count = shop_mwifi_count[shop_mwifi_count['rank'] < 3]  # 选择最强bssid出现次数较多的店铺
        result = candidate_temp.merge(shop_mwifi_count, on='bssid', how='inner')[['row_id', 'shop_id']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户去过的商店
def get_people_sample(data, candidate, data_key):
    result_path = cache_path + 'people_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_shop_count = data.groupby(['user_id', 'shop_id'], as_index=False)['user_id'].agg(
            {'user_shop_count': 'count'})  # data数据集user  shop的统计
        result = candidate.merge(user_shop_count, on='user_id', how='inner')
        result = result[['row_id', 'shop_id']]  # 只返回row_id和shop_id
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 根据坐标添加前三
def get_loc_sample(data, candidate, data_key):
    result_path = cache_path + 'loc_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data['cwifi'] = data['wifi_infos'].apply(get_connect_wifi)  # 没有连接的为nan
        data = data[~data['cwifi'].isnull()]  # 非nan的数据，也就是有连接数据
        shop_loc_dict = {}
        for shop_id, grp in data.groupby('shop_id'):  # 按shop_id分组
            locs = []
            for row in grp.itertuples():
                locs.append((row.longitude, row.latitude))  # 有连接店铺的经纬度
            shop_loc_dict[shop_id] = locs  # shop_id:[(longitude,latitude),()]# 通过连接wifi进行选取，可能是连接wifi获取了经纬比基站更准确
        shop = pd.read_csv(shop_path)
        mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()

        def loc_knn2_loss(shop_id, longitude, latitude):
            loss = 0
            try:
                locs = shop_loc_dict[shop_id]  # shop_id的经纬集合（非店铺自身的经纬）
                for (lon, lat) in locs:
                    loss += 0.1 ** (((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5 * 100000)  # 离得越远loss越小
            except:
                loss = np.nan
            return loss

        result = []
        candidate_temp = candidate.copy()
        candidate_temp['cwifi'] = candidate_temp['wifi_infos'].apply(get_connect_wifi)
        for row in candidate_temp.itertuples():
            if row.cwifi is np.nan:  # 如果样本没有连接的wifi
                longitude = row.longitude  # 样本的经纬
                latitude = row.latitude
                for shop_id in mall_shop_dict[row.mall_id]:
                    result.append(
                        [row.row_id, shop_id, loc_knn2_loss(shop_id, longitude, latitude)])  # 改样本的经纬和该店铺连接wifi的经纬计算loss
        result = pd.DataFrame(result, columns=['row_id', 'shop_id', 'loc_knn2'])  # 包含了同mall里所有其他的店铺loss
        result = group_rank(result, 'row_id', 'loc_knn2', ascending=False)  # 降序
        result = result[result['rank'] < 3][['row_id', 'shop_id']]  # 选择loss较大的店铺
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_cos_sample(data, candidate, data_key):
    result_path = cache_path + 'cos_sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_mean_signal_dict = defaultdict(lambda: -104)
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, signal])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi['signal'] = shop_wifi['signal'].astype('int')
        shop_wifi_mean_signal_dict.update(shop_wifi.groupby(['shop_id', 'bssid'])['signal'].mean().to_dict())

        def signal_weight(signal):
            return 1.06 ** (signal)
        def knn_loss(shop_id, wifis):
            loss = 0; wlen = 0; slen = 0
            for bssid in wifis:
                a = signal_weight(wifis[bssid])
                b = signal_weight(shop_wifi_mean_signal_dict[(shop_id, bssid)])
                loss += a * b
                slen += a ** 2
                wlen += b ** 2
            loss_cos = loss / (slen ** 0.5 * wlen ** 0.5)
            return loss_cos
        shop = pd.read_csv(shop_path)######
        mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()#############
        result = []
        for row in candidate.itertuples():

            if row.wifi_infos != '':
                shops = mall_shop_dict[row.mall_id]
                wifis = {}
                for wifi_info in row.wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifis[bssid] = int(signal)
                for shop_id in shops:
                    result.append([row.row_id,shop_id,knn_loss(shop_id, wifis)])

        result = pd.DataFrame(result, columns=['row_id','shop_id','cos'])
        result = group_rank(result,'row_id','cos',ascending=False)#cos越大应该月靠前,降序
        result = result[result['rank']<3][['row_id', 'shop_id']]#保留前三
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

###################################################
# ..................... 构造特征 ....................
###################################################
# 连接wifi特征
def get_cwifi_feat(data, sample, data_key):
    result_path = cache_path + 'cwifi_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sample_temp = sample.copy()
        sample_temp['bssid'] = sample_temp['wifi_infos'].apply(get_connect_wifi)  # 连接的bssid
        shop_cwifi_count = get_shop_cwifi_count(data, data_key)  # bssid shop 的计数
        cwifi_count = shop_cwifi_count.groupby('bssid', as_index=False)['shop_cwifi_count'].agg(
            {'cwifi_count': 'sum'})  # 按bssid分组计数
        shop_cwifi_count = shop_cwifi_count.merge(cwifi_count, on='bssid', how='left')
        shop_cwifi_count['shop_cwifi_rate'] = shop_cwifi_count['shop_cwifi_count'] / shop_cwifi_count[
            'cwifi_count']  # 计算比率
        result = sample_temp.merge(shop_cwifi_count, on=['shop_id', 'bssid'], how='left')
        result = result[['shop_cwifi_count', 'cwifi_count', 'shop_cwifi_rate']].fillna(0)  # 三种统计，没有的用0填充
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# tfidf特征
def get_tfidf_feat(data, sample, data_key):
    result_path = cache_path + 'tfidf_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_tfidf = get_shop_wifi_tfidf(data, data_key)
        shop_wifi_tfidf_dict = {}
        for shop_id, grp in shop_wifi_tfidf.groupby('shop_id'):
            wifi_idf = {}
            for tuple in grp.itertuples():
                wifi_idf[tuple.bssid] = tuple.tfidf
                shop_wifi_tfidf_dict[shop_id] = wifi_idf
        result = []
        row_tfidf_dict = defaultdict(lambda: 0)
        for row in sample.itertuples():
            wifi_infos = row.wifi_infos
            shop_id = row.shop_id
            tfidf = 0
            if wifi_infos != '':
                wifi_infos = [wifi.split('|') for wifi in wifi_infos.split(';')]
                for i, (bssid, signal, Flag) in enumerate(wifi_infos):
                    try:
                        tfidf += shop_wifi_tfidf_dict[shop_id][bssid] * rank_weight(i)
                    except:
                        pass
            result.append([row.row_id, tfidf])
            row_tfidf_dict[row.row_id] += tfidf
        result = pd.DataFrame(result, columns=['row_id', 'tfidf'])
        result['row_tfidf'] = result['row_id'].map(row_tfidf_dict)
        result = result[['tfidf', 'row_tfidf']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用最强信号选取前二样本
def get_mwifi_feat(data, sample, data_key):
    result_path = cache_path + 'mwifi_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sample_temp = sample.copy()
        sample_temp['bssid'] = sample_temp['wifi_infos'].apply(get_most_wifi)
        shop_mwifi_count = get_shop_mwifi_count(data, data_key)
        mwifi_count = shop_mwifi_count.groupby('bssid', as_index=False)['shop_mwifi_count'].agg({'mwifi_count': 'sum'})
        shop_mwifi_count = shop_mwifi_count.merge(mwifi_count, on='bssid', how='left')
        shop_mwifi_count['shop_mwifi_rate'] = shop_mwifi_count['shop_mwifi_count'] / shop_mwifi_count['mwifi_count']
        result = sample_temp.merge(shop_mwifi_count, on=['shop_id', 'bssid'], how='left')
        result = result[['shop_mwifi_count', 'mwifi_count', 'shop_mwifi_rate']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# knn
def get_knn_feat(data, sample, data_key):
    result_path = cache_path + 'knn_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_mean_signal_dict = defaultdict(lambda: -104)
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, signal])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi['signal'] = shop_wifi['signal'].astype('int')
        shop_wifi_mean_signal_dict.update(shop_wifi.groupby(['shop_id', 'bssid'])['signal'].mean().to_dict())

        def signal_weight(signal):
            return 1.014 ** (signal)

        def knn_loss(shop_id, wifis):
            loss = 0
            for bssid in wifis:
                a = signal_weight(wifis[bssid])
                b = signal_weight(shop_wifi_mean_signal_dict[(shop_id, bssid)])
                loss += (a - b) ** 2
            return loss

        result = []
        for row in sample.itertuples():
            shop_id = row.shop_id
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_infos in row.wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_infos.split('|')
                    wifis[bssid] = int(signal)
            result.append([knn_loss(shop_id, wifis)])
        result = pd.DataFrame(result, columns=['knn'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# wifi余弦相似度
def get_cos_feat(data, sample, data_key):
    result_path = cache_path + 'cos_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_mean_signal_dict = defaultdict(lambda: -104)
        shop_wifi = []
        for row_id, ship_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    shop_wifi.append([row_id, ship_id, bssid, signal])
        shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
        shop_wifi['signal'] = shop_wifi['signal'].astype('int')
        shop_wifi_mean_signal_dict.update(shop_wifi.groupby(['shop_id', 'bssid'])['signal'].mean().to_dict())

        def signal_weight(signal):
            return 1.06 ** (signal)

        def knn_loss(shop_id, wifis):
            loss = 0;
            wlen = 0;
            slen = 0
            for bssid in wifis:
                a = signal_weight(wifis[bssid])
                b = signal_weight(shop_wifi_mean_signal_dict[(shop_id, bssid)])
                loss += a * b
                slen += a ** 2
                wlen += b ** 2
            loss_cos = loss / (slen ** 0.5 * wlen ** 0.5)
            return loss_cos

        result = []
        for row in sample.itertuples():
            shop_id = row.shop_id
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_infos in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_infos.split('|')
                    wifis[bssid] = int(signal)
                result.append(knn_loss(shop_id, wifis))
            else:
                result.append(np.nan)
        result = pd.DataFrame(result, columns=['cos'])
        mean_cos = result['cos'].values.mean()
        result.fillna(mean_cos, inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# knn2
def get_knn2_feat(data, sample, data_key):
    result_path = cache_path + 'knn2_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_wifi_signal_dict = defaultdict(lambda: [])
        for shop_id, wifi_infos in zip(data['shop_id'].values, data['wifi_infos'].values):
            if wifi_infos != '':
                wifi_signal_dict = defaultdict(lambda: -104)
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifi_signal_dict[bssid] = int(signal)
                shop_wifi_signal_dict[shop_id].append(wifi_signal_dict)
        row_shop_dict = defaultdict(lambda: [])
        row_shop_dict.update(sample.groupby('row_id')['shop_id'].unique().to_dict())

        def signal_weight(signal):
            return 1.014 ** (signal)

        def knn2_loss(shop_id, wifis):
            loss = 0
            for dt_wifis in shop_wifi_signal_dict[shop_id]:
                single_loss = 0
                for bssid in wifis:
                    a = signal_weight(wifis[bssid])
                    b = signal_weight(dt_wifis[bssid])
                    single_loss += (a - b) ** 2
                loss += (2 * 10 ** 49) ** (0 - single_loss)
            return loss

        result = []
        for row in sample.itertuples():
            shop_id = row.shop_id
            wifis = {}
            wifi_infos = row.wifi_infos
            if wifi_infos != '':
                for wifi_infos in row.wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_infos.split('|')
                    wifis[bssid] = int(signal)
            result.append([knn2_loss(shop_id, wifis)])
        result = pd.DataFrame(result, columns=['knn2'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        print('done')
    return result


# 商店出现的次数
def get_shop_count(data, sample, data_key):
    result_path = cache_path + 'shop_count_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_count = data.groupby('shop_id', as_index=False)['user_id'].agg({'shop_count': 'count',
                                                                             'shop_n_user': 'nunique'})
        result = sample.merge(shop_count, on='shop_id', how='left')[['shop_count', 'shop_n_user']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户去过此商店的次数
def get_user_shop_count(data, sample, data_key):
    result_path = cache_path + 'user_shop_count_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_count = data.groupby(['user_id'], as_index=False)['user_id'].agg(
            {'user_count': 'count'})
        user_shop_count = data.groupby(['user_id', 'shop_id'], as_index=False)['user_id'].agg(
            {'user_shop_count': 'count'})
        result = sample.merge(user_count, on=['user_id'], how='left')
        result = result.merge(user_shop_count, on=['user_id', 'shop_id'], how='left')
        result['user_shop_rate'] = result['user_shop_count'] / result['user_count']
        result = result[['user_shop_count', 'user_count', 'user_shop_rate']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 经纬度knn2
def get_loc_knn2_feat(data, sample, data_key):
    result_path = cache_path + 'loc_knn2_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        shop_loc_dict = {}
        for shop_id, grp in data.groupby('shop_id'):
            locs = []
            for row in grp.itertuples():
                locs.append((row.longitude, row.latitude))
            shop_loc_dict[shop_id] = locs

        def loc_knn2_loss(shop_id, longitude, latitude):
            loss = []
            try:
                locs = shop_loc_dict[shop_id]
                for (lon, lat) in locs:
                    loss.append(0.1 ** (((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5 * 100000))
            except:
                pass
            return (sum(loss), np.mean(loss), np.median(loss), np.min(loss), np.max(loss))

        result = []
        for row in sample.itertuples():
            longitude = row.longitude;
            latitude = row.latitude
            result.append(loc_knn2_loss(row.shop_id, longitude, latitude))
        result = pd.DataFrame(result,
                              columns=['loc_knn2', 'loc_knn2_mean', 'loc_knn2_median', 'loc_knn2_min', 'loc_knn2_max'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 距离店铺中心的距离
def get_loc_knn_feat(data, sample, data_key):
    result_path = cache_path + 'loc_knn_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data['cwifi'] = data['wifi_infos'].apply(get_connect_wifi)
        data = data[~data['cwifi'].isnull()]
        shop_loc_mean_dict = data[['shop_id', 'longitude', 'latitude']].groupby('shop_id').mean()
        shop_loc_mean_dict = dict(zip(shop_loc_mean_dict.index, shop_loc_mean_dict.values))

        def loc_knn_loss(shop_id, longitude, latitude):
            loss = 0
            try:
                lon, lat = shop_loc_mean_dict[shop_id]
                loss += ((lon - longitude) ** 2 + (lat - latitude) ** 2) ** 0.5
            except:
                loss = np.nan
            return loss

        result = []
        sample_temp = sample.copy()
        sample_temp['cwifi'] = sample_temp['wifi_infos'].apply(get_connect_wifi)
        for row in sample_temp.itertuples():
            if row.cwifi is np.nan:
                longitude = row.longitude
                latitude = row.latitude
                result.append(loc_knn_loss(row.shop_id, longitude, latitude))
            else:
                result.append(np.nan)
        result = pd.DataFrame(result, columns=['loc_knn'])
        mean_loc_knn = result['loc_knn'].values.mean()
        result.fillna(mean_loc_knn, inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户曾在该类型的店铺消费过
def get_user_has_been_shop_feat(data, sample, data_key):
    result_path = cache_path + 'user_has_been_shop_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        sample_temp = sample.copy()
        data_temp = data_temp.merge(shop[['shop_id', 'category_id']], how='left', on='shop_id')  # 合并店铺类别
        sample_temp = sample_temp.merge(shop[['shop_id', 'category_id']], how='left', on='shop_id')
        user_category = data_temp.groupby(['user_id', 'category_id'], as_index=False)['user_id'].agg(
            {'user_category_count': 'count'})
        user_count = data_temp.groupby(['user_id'], as_index=False)['user_id'].agg({'user_count': 'count'})
        result = sample_temp.merge(user_count, on=['user_id'], how='left')
        result = result.merge(user_category, on=['user_id', 'category_id'], how='left')
        result['user_category_rate'] = result['user_category_count'] / result['user_count']
        result = result[['user_category_count', 'user_category_rate']].fillna(0)  # user_count已经统计过
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户历史平均消费指数的与店铺消费指数做差
def get_user_price_feat(data, sample, data_key):
    result_path = cache_path + 'user_price_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp.merge(shop[['shop_id', 'price']], how='left', on='shop_id')
        sample_temp = sample.copy()
        sample_temp = sample_temp.merge(shop[['shop_id', 'price']], how='left', on='shop_id')
        user_price = data_temp.groupby(['user_id', 'price'], as_index=False)['price']. \
            agg({'user_mean_price': 'mean', 'user_max_price': 'max', 'user_min_price': 'min'})
        result = sample_temp.merge(user_price[['user_id', 'user_mean_price', 'user_max_price', 'user_min_price']],
                                   on=['user_id'], how='left')
        result['user_mean_shop'] = result['user_mean_price'] - result['price']
        #        result['user_max_shop']=result['user_max_price']-result['price']
        #        result['user_min_shop']=result['price']-result['user_min_price']
        user_mean_shop_mean = result['user_mean_shop'].mean()
        #        user_max_shop_mean=result['user_max_shop'].values.mean()
        #        user_min_shop_mean=result['user_min_shop'].values.mean()
        result['user_mean_shop'].fillna(user_mean_shop_mean, inplace=True)
        #        result['user_max_shop'].fillna(user_max_shop_mean,inplace=True)
        #        result['user_min_shop'].fillna(user_min_shop_mean,inplace=True)
        #        result=result[['user_mean_shop','user_max_shop','user_min_shop']]
        result = result[['user_mean_shop']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 同一小时时间店铺的统计
def get_time_count_feat(data, sample, data_key):
    flag = False
    result_path = cache_path + 'time_count_{}.pickle'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp['time_stamp'] = pd.to_datetime(data_temp['time_stamp'])
        data_temp['hour'] = data_temp.time_stamp.apply(lambda x: x.hour)
        data_temp['week_day'] = data_temp.time_stamp.apply(lambda x: x.weekday())
        data_temp['is_weekend'] = data_temp['week_day'].apply(lambda x: int(x >= 6))
        shop_hour_count = data_temp.groupby(['shop_id', 'hour'], as_index=False)['shop_id'].agg(
            {'shop_hour_count': 'count'})
        shop_hour_min_max = data_temp.groupby(['shop_id'], as_index=False)['hour'].agg(
            {'shop_hour_min': 'min', 'shop_hour_max': 'max'})
        sample_temp = sample.copy()
        shop_weekend_count = data_temp.groupby(['shop_id', 'is_weekend'], as_index=False)['shop_id'].agg(
            {'shop_weekend_count': 'count'})
        sample_temp['time_stamp'] = pd.to_datetime(sample_temp['time_stamp'])
        sample_temp['hour'] = sample_temp.time_stamp.apply(lambda x: x.hour)
        sample_temp['week_day'] = sample_temp.time_stamp.apply(lambda x: x.weekday())
        sample_temp['is_weekend'] = sample_temp['week_day'].apply(lambda x: int(x >= 6))
        result = sample_temp.merge(shop_hour_count, on=['shop_id', 'hour'], how='left')
        result = result.merge(shop_hour_min_max, on='shop_id', how='left')
        result = result.merge(shop_weekend_count, on=['shop_id', 'is_weekend'], how='left')
        result['is_in_business'] = (
        (result['hour'] <= result['shop_hour_max']) & (result['hour'] >= result['shop_hour_min'])).astype('int').fillna(
            -1)
        result = result[['shop_hour_count', 'shop_weekend_count', 'is_in_business']].fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# wifi rank 与店铺wifi  rank ，求l1和l2距离
def get_wifi_rank_l1_l2_feat(data, sample, data_key):
    result_path = cache_path + 'wifi_rank_l1_l2_{}.pickle'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        wifi_rank_list = []
        for row_id, shop_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                rank = 0
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifi_rank_list.append([shop_id, bssid, rank])
                    rank += 1
        wifi_rank = pd.DataFrame(wifi_rank_list, columns=['shop_id', 'bssid', 'rank'])
        wifi_rank_dict = defaultdict(lambda: 10)
        wifi_rank_dict.update(wifi_rank.groupby(['shop_id', 'bssid'])['rank'].agg('mean').to_dict())
        result = []

        def get_wifi_rank_l1_l2(shop_id, wifi_infos):
            rank = 0
            wifi_rank_l1 = 0
            wifi_rank_l2 = 0
            if wifi_infos == '':
                return (np.nan, np.nan)
            for wifi_info in wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')

                #                if wifi_rank_dict[(shop_id,wifi_info)]!=11:
                #                wifi_rank_co_len+=1
                wifi_rank_l1 += abs(wifi_rank_dict[(shop_id, bssid)] - rank)
                wifi_rank_l2 += (wifi_rank_dict[(shop_id, bssid)] - rank) ** 2
                rank += 1
            return (wifi_rank_l1, math.sqrt(wifi_rank_l2))

        for row_id, shop_id, wifi_infos in zip(sample['row_id'].values, sample['shop_id'].values,
                                               sample['wifi_infos'].values):
            wifi_rank_l1, wifi_rank_l2 = get_wifi_rank_l1_l2(shop_id, wifi_infos)
            result.append([wifi_rank_l1, wifi_rank_l2])

        result = pd.DataFrame(result, columns=['wifi_rank_l1', 'wifi_rank_l2'])
        wifi_rank_l1_mean = result.wifi_rank_l1.mean()
        wifi_rank_l2_mean = result.wifi_rank_l2.mean()
        result['wifi_rank_l1'] = result['wifi_rank_l1'].fillna(wifi_rank_l1_mean)
        result['wifi_rank_l2'] = result['wifi_rank_l2'].fillna(wifi_rank_l2_mean)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 统计wifi在店铺历史wifi中出现的的相关统计
def get_wifi_shop_count_feat(data, sample, data_key):
    result_path = cache_path + 'wifi_shop_count_{}.pickle'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        wifi_shop_dict = defaultdict(lambda: 0)
        for row_id, shop_id, wifi_infos in zip(data['row_id'].values, data['shop_id'].values,
                                               data['wifi_infos'].values):
            if wifi_infos != '':
                for wifi_info in wifi_infos.split(';'):
                    bssid, signal, Flag = wifi_info.split('|')
                    wifi_shop_dict[(bssid, shop_id)] += 1

        # wifi_shop=pd.DataFrame(wifi_shop_list,columns=['shop_id','bssid'])
        # wifi_shop=wifi_shop.groupby(['shop_id','bssid'],as_index=False)['shop_id'].agg({'wifi_shop_count':'count'})
        # data_temp=data.copy()
        shop_count = data.groupby(['shop_id'], as_index=False)['shop_id'].agg({'shop_count': 'count'})
        sample_temp = sample.copy()

        def get_wifi_shop_count(shop_id, wifi_infos):
            if wifi_infos == '':
                return 0
            wifi_count = 0
            for wifi_info in wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                wifi_count += wifi_shop_dict[(bssid, shop_id)]
            return wifi_count

        wifi_count_sum = []

        for row_id, shop_id, wifi_infos in zip(sample['row_id'].values, sample['shop_id'].values,
                                               sample['wifi_infos'].values):
            wifi_count_sum.append(get_wifi_shop_count(shop_id, wifi_infos))
        result = pd.DataFrame(wifi_count_sum, columns=['wifi_count_sum'])
        print(sample_temp.shape, result.shape)
        result = pd.concat([sample_temp, result], axis=1)
        result = result.merge(shop_count, on='shop_id', how='left')
        result['wifi_count_sum_rate'] = result['wifi_count_sum'] / result['shop_count']
        result = result.fillna(0)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 二次处理特征
def second_feat(result):
    return result


# 构造样本
def get_sample(data, candidate, data_key):
    result_path = cache_path + 'sample_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        cwifi_sample = get_cwifi_sample(data, candidate, data_key)  # 用连接的wifi选取样本     #是data和candidate共同选出的候选集，应该适用于两者
        tfidf_sample = get_tfidf_sample(data, candidate, data_key)  # 用简单knn选取前3样本
        mwifi_sample = get_mwifi_sample(data, candidate, data_key)  # 用最强信号选取前二样本
        people_sample = get_people_sample(data, candidate, data_key)  # 用户去过的商店（全部）
        loc_sample = get_loc_sample(data, candidate, data_key)  # 根据坐标添加前二

        # 汇总样本id
        result = pd.concat([cwifi_sample,
                            tfidf_sample,
                            mwifi_sample,
                            people_sample,
                            loc_sample]).drop_duplicates()  # 构造出了row_id对应的shop_id,有可能重复，需去重
        # 剔除错误样本
        shop = pd.read_csv(shop_path)
        shop_mall_dict = dict(zip(shop['shop_id'].values, shop['mall_id']))  # shop mall信息构造对
        # result只是构造出的row_id和shop_id，所以要合并其他信息
        result = result.merge(
            candidate[['row_id', 'user_id', 'mall_id', 'longitude', 'latitude', 'time_stamp', 'wifi_infos']],
            on='row_id', how='left')
        result = result[result['shop_id'].map(shop_mall_dict) == result['mall_id']]
        result.index = list(range(result.shape[0]))
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('样本个数为：{}'.format(result.shape))
    return result


# 制作训练集
def make_feats(data, candidate):  # train test
    t0 = time.time()
    data_key = hashlib.md5(data['time_stamp'].to_string().encode()).hexdigest() + \
               hashlib.md5(candidate['time_stamp'].to_string().encode()).hexdigest()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'train_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        print('清洗WiFi...')
        data, candidate = clear(data, candidate, data_key)  # 清楚wifi_infos中wifi出现次数小于阈值的wifi
        # row_wifi_infos_dict = get_row_wifi_infos_dict(data, candidate,data_key)
        print('开始构造样本...')
        sample = get_sample(data, candidate, data_key)  # 构造样本
        print('样本大小', sample.shape)
        print('开始构造特征...')

        cwifi_feat = get_cwifi_feat(data, sample, data_key)  # 连接的wifi个数
        tfidf_feat = get_tfidf_feat(data, sample, data_key)  # tfidf
        mwifi_feat = get_mwifi_feat(data, sample, data_key)  # 最强信号个数
        knn_feat = get_knn_feat(data, sample, data_key)  # knn
        cos_feat = get_cos_feat(data, sample, data_key)  # wifi余弦相似度
        knn2_feat = get_knn2_feat(data, sample, data_key)  # knn2
        # 从单个wifi角度来判断
        # 用auc从相对大小来判断
        # 用户是否连接过wifi与商店对应的最强wifi是否有过交集
        shop_count = get_shop_count(data, sample, data_key)  # 商店出现的次数
        user_shop_count = get_user_shop_count(data, sample, data_key)  # 用户去过此商店的次数
        # 用户是否去过此种类型的商店
        loc_knn2_feat = get_loc_knn2_feat(data, sample, data_key)  # 经纬度knn2
        loc_knn_feat = get_loc_knn_feat(data, sample, data_key)  # 距离店铺中心的距离
        user_has_been_shop_feat = get_user_has_been_shop_feat(data, sample, data_key)

        user_price_feat = get_user_price_feat(data, sample, data_key)
        time_count_feat = get_time_count_feat(data, sample, data_key)
        wifi_rank_l1_l2_feat = get_wifi_rank_l1_l2_feat(data, sample, data_key)
        wifi_shop_count_feat = get_wifi_shop_count_feat(data, sample, data_key)
        print('开始合并特征...')
        result = concat([sample, cwifi_feat, tfidf_feat, mwifi_feat, knn_feat, cos_feat,
                         shop_count, user_shop_count, knn2_feat, loc_knn2_feat,
                         loc_knn_feat, user_has_been_shop_feat, user_price_feat,
                         time_count_feat, wifi_rank_l1_l2_feat, wifi_shop_count_feat])
        result = second_feat(result)

        # 线下测评添加label
        #        print('添加label')
        #        result = get_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result


# from mayi.feat1 import *
import lightgbm as lgb

# test = test.sample(frac=0.1,random_state=66, axis=0)

test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id', 'mall_id']], on='shop_id', how='left')
t1 = train[train.time_stamp < '2017-08-25']  # 训练集提取特征
t2 = train[train.time_stamp >= '2017-08-25'].drop('shop_id', axis=1)  # 训练集
t3 = train  # 测试集提取特征
t4 = test  # 测试集

# 此处可以只运行一次保存即可，
d_true = {}
for i in train.itertuples():
    d_true[i.row_id] = i.shop_id
with open(data_path + 'true.pkl', 'wb') as f:
    pickle.dump(d_true, f)
del d_true

# 对wifi排序
train_wifi_infos = []
for wifi_info in t1['wifi_infos'].values:
    train_wifi_infos.append(
        ';'.join(sorted([wifi for wifi in wifi_info.split(';')], key=lambda x: int(x.split('|')[1]), reverse=True)))
t1['wifi_infos'] = train_wifi_infos

test_wifi_infos = []
for wifi_info in t2['wifi_infos'].values:
    test_wifi_infos.append(
        ';'.join(sorted([wifi for wifi in wifi_info.split(';')], key=lambda x: int(x.split('|')[1]), reverse=True)))
t2['wifi_infos'] = test_wifi_infos
train_wifi_infos3 = []
for wifi_info in t3['wifi_infos'].values:
    train_wifi_infos3.append(
        ';'.join(sorted([wifi for wifi in wifi_info.split(';')], key=lambda x: int(x.split('|')[1]), reverse=True)))
t3['wifi_infos'] = train_wifi_infos3

test_wifi_infos4 = []
for wifi_info in t4['wifi_infos'].values:
    test_wifi_infos4.append(
        ';'.join(sorted([wifi for wifi in wifi_info.split(';')], key=lambda x: int(x.split('|')[1]), reverse=True)))
t4['wifi_infos'] = test_wifi_infos4

data_train = make_feats(t1, t2).fillna(0)  # 训练集构造特征
data_test = make_feats(t3, t4).fillna(0)  # 测试集构造特征

data_train = grp_normalize(data_train, 'row_id', ['knn', 'knn2'], start=0)
data_train = grp_rank(data_train, 'row_id', ['cos'], ascending=False)
data_train = grp_standard(data_train, 'row_id', ['shop_count', 'loc_knn2',
                                                 'shop_hour_count', 'shop_weekend_count',
                                                 'wifi_count_sum'])
data_train = get_label(data_train)  # 添加label

data_test = grp_normalize(data_test, 'row_id', ['knn', 'knn2'], start=0)
data_test = grp_rank(data_test, 'row_id', ['cos'], ascending=False)
data_test = grp_standard(data_test, 'row_id', ['shop_count', 'loc_knn2',
                                               'shop_hour_count', 'shop_weekend_count',
                                               'wifi_count_sum'])

predictors = data_train.columns.drop(['row_id', 'time_stamp', 'user_id', 'shop_id',
                                      'mall_id', 'wifi_infos', 'label', 'is_in_business',
                                      'wifi_rank_l2', 'shop_weekend_count', 'wifi_count_sum'])

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 66,
}

# 线下验证
# train_train=data_train[data_train.time_stamp<'2017-08-29']        #线下训练集
# train_val=data_train[data_train.time_stamp>='2017-08-29']             #线下验证集
# lgb_train = lgb.Dataset(train_train[predictors], train_train.label)
# lgb_test = lgb.Dataset(train_val[predictors], train_val.label,reference=lgb_train)
# gbm = lgb.train(params,
#                lgb_train,
#                num_boost_round=10000,
#                valid_sets={lgb_train,lgb_test},
#                verbose_eval = 50,
#                early_stopping_rounds=200)
# feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
# feat_imp.to_csv(cache_path + 'feat_imp.csv')
# preds = gbm.predict(train_val[predictors])
# train_val['pred'] = preds
# train_val_temp=train_val.sort_values('pred')
# result = train_val_temp.drop_duplicates('row_id',keep='last')
# t = t2[t2.time_stamp>='2017-08-29']
# t = t.merge(result,on='row_id',how='left')
# print('准确率为：{}'.format(acc(t)))

lgb_train = lgb.Dataset(data_train[predictors], data_train.label)
gbm = lgb.train(params, lgb_train, num_boost_round=1100, valid_sets=lgb_train, verbose_eval=50)
pred = gbm.predict(data_test[predictors])
data_test['pred'] = pred
data_test_temp = data_test.sort_values('pred')
result = data_test_temp.drop_duplicates('row_id', keep='last')
result = result[['row_id', 'shop_id']]
# 结果缺失，进行填补
tmp = pd.DataFrame(columns=['row_id'], data=list(set(test.row_id) - set(result.row_id)))
tmp['shop_id'] = 's_100'
result = pd.concat([result, tmp])
result.to_csv(data_path + 'result.csv', index=False)
