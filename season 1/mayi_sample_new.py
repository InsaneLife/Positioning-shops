# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:08:28 2017

@author: rjgcyjs
"""
'''
单独测试
cwifi_sample 0.167910
tfidf_sample  0.92218
mwifi_sample0.9056
people_sample  0.2026
loc_sample   0.6933
knn_sample     0.94499
cos_sample  0.88059
loc_knn2_sample 0.86249

由于loc_knn2_sample较为费时间，对此单独对比
全部sample 0.982
单独排除loc_knn2_sample  0.980

单独排除loc_sample  0.9807
'''


def get_knn_sample(data,candidate,data_key):
    result_path = cache_path + 'knn_sample_{}.hdf'.format(data_key)
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
                loss += (a-b)**2
            return loss
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
        result = pd.DataFrame(result,columns=['row_id','shop_id','knn'])
        result = group_rank(result,'row_id','knn',ascending=True)#knn升序，小的靠前
        result = result[result['rank']<3][['row_id', 'shop_id']]#保留前三
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

def get_cos_sample(data,candidate,data_key):
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

def get_loc_knn2_sample(data,candidate,data_key):
    result_path = cache_path + 'loc_knn2_sample_{}.hdf'.format(data_key)
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
            return np.mean(loss)
        shop = pd.read_csv(shop_path)######
        mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()#############
        result = []
        for row in candidate.itertuples():
            longitude = row.longitude;
            latitude = row.latitude
            shops = mall_shop_dict[row.mall_id]
            for shop_id in shops:
                if shop_id in shop_loc_dict:
                    result.append([row.row_id,shop_id,loc_knn2_loss(shop_id, longitude, latitude)])
        result = pd.DataFrame(result, columns=['row_id','shop_id','loc_knn2'])
        result = group_rank(result,'row_id','loc_knn2',ascending=False)#loc_knn2越小应该月靠前,降序
        result = result[result['rank']<3][['row_id', 'shop_id']]#保留前三
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result