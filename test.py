import torch.nn.functional as F
import torch
a=torch.Tensor([0.1])
b=torch.Tensor([1])
print(F.binary_cross_entropy_with_logits(a,b))

# ## 查看生成数据分布
# import os

# import matplotlib.pyplot as plt

# path = '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_11'
# files = os.listdir(path)

# print(len(files))

# import pickle

# X = []
# Y = []
# objList = []
# for file_name in files:
#     if not file_name.endswith('.pkl'):
#         continue
#     # 构造完整的文件路径
#     file_path = os.path.join(path, file_name)
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#         X.append(data['objList'][0][1])
#         Y.append(data['objList'][0][2])
#         objList.append(data['objList'])

# plt.scatter(X,Y)
# plt.savefig('test.png')
# with open('locs.pkl','wb') as f:
#     pickle.dump(objList,f)