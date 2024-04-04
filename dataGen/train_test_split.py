import os
import shutil
from random import sample

# 设置源目录和目标目录
source_directory = '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_graspTargetObj_Right_0402'
target_directory = '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_graspTargetObj_Right_test'

# 获取源目录中所有文件夹的列表
folders = [f for f in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, f))]

# 计算要移动的文件夹数量：20% 的文件夹
number_to_move = int(len(folders) * 0.2)

# 随机选择要移动的文件夹
folders_to_move = sample(folders, number_to_move)

# 移动选定的文件夹
for folder in folders_to_move:
    shutil.move(os.path.join(source_directory, folder), target_directory)

# # 打印出被移动的文件夹
# print(f"Moved folders: {folders_to_move}")
