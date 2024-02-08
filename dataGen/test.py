import pickle
with open('/data2/liangxiwen/RM-PRT/0718_single_merge_data_new/000001.pkl','rb') as f:
    data=pickle.load(f)
print(data['trajectory'][0]['img'].shape)