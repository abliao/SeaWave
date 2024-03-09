import pickle

import os

def list_all_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                files_list.append(os.path.join(root, file))
    return files_list

# 调用函数
directory = "/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_knockOver_Right"
all_files = list_all_files(directory)
for file in all_files:
    with open(file,'rb') as f:
        data=pickle.load(f)
    data['event'] = 'knockOver'
    with open(file,'wb') as f:
        pickle.dump(data,f)