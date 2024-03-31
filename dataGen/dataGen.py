#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import sys
import time
import json

sys.path.append('./')
sys.path.append('../')
from src.Env.simUtils import *
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip, ImageClip, concatenate_videoclips

import grpc
import argparse
import pickle
import os

import hashlib
import random

def string_to_seed(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    # 将哈希值转换为整数
    seed = int(hex_dig, 16)%100000009
    return seed


# 创建解析器
parser = argparse.ArgumentParser(description="数据生成")

# 添加参数
parser.add_argument('--host', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--data_info', type=str, default='')
parser.add_argument('--n_objs', type=int)
parser.add_argument('--handSide', type=str)
parser.add_argument('--event', type=str)
# 解析命令行参数
args = parser.parse_args()

print('数据生成脚本启动')
print("端口为:",args.host)
host = args.host
scene_num = 1
map_id = 2
server = SimServer(host, scene_num=scene_num, map_id=map_id)
sim = SimAction(host, scene_id=0)

# # 生成种子
# seed = string_to_seed(args.output_path)
# random.seed(seed)
# np.random.seed(seed)

events = {
    'graspTargetObj':{'act':sim.graspTargetObj,'check':sim.checkGraspTargetObj},
    'placeTargetObj':{'act':sim.placeTargetObj,'check':sim.checkPlaceTargetObj},
    'moveNear':{'act':sim.moveNear,'check':sim.checkMoveNear},
    'knockOver':{'act':sim.knockOver,'check':sim.checkKnockOver},
    'pushFront':{'act':sim.pushFront,'check':sim.checkPushFront},
    'pushLeft':{'act':sim.pushLeft,'check':sim.checkPushLeft},
    'pushRight':{'act':sim.pushRight,'check':sim.checkPushRight}
}
event = events[args.event]


output_path = '/data2/liangxiwen/zkd/datasets/dataGen/DATA'+ os.sep+args.output_path
data_info=args.data_info
meta_data_path = output_path + os.sep + 'meta_data.json'
n_objs = args.n_objs
handSide = args.handSide

can_list = list(sim.can_list)
if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(meta_data_path):
    meta_data = {
        "collected_num": 0,
        "start_index": 0,
        "info": data_info,
    }
    # 将数据转换成JSON格式
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f)
else:
    with open(meta_data_path, 'rb') as f:
        meta_data = json.load(f)

log_images_path = './log_images'
if not os.path.exists(log_images_path):
    os.makedirs(log_images_path)
grasp_images_path = './grasp_images'
if not os.path.exists(grasp_images_path):
    os.makedirs(grasp_images_path)
before_grasp_images_path = './before_grasp_images'
if not os.path.exists(before_grasp_images_path):
    os.makedirs(before_grasp_images_path)
video_path = './log_videos'
if not os.path.exists(video_path):
    os.makedirs(video_path)

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def Resize(mat):
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize((224, 224))
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat / 255.0
    return mat


from tqdm import tqdm

import re
# f=open('Imitation_data/RLexpert/0816_two_obj_data.txt')
f=open('Imitation_data/RLexpert/0718_single_merge_data.txt')
data=[]
for line in f.readlines():
    line = line.strip('\n')
    data.append(line)
datas=[]
last_index=0
for i in range(len(data)):
    if data[i]=='':
        datas.append(data[last_index:i])
        last_index=i+1
df=[]
for i in datas:
    data=[]
    for j in i:
        result = re.split(',|;', j)
        numbers=list(map(float, result))
        data.append(numbers)
    df.append(data)
print('load data over')


collected_num = meta_data['collected_num']
start_index = meta_data['start_index']
for epoch in range(1):
    print('Epoch:', epoch)
    offline_data = dict()
    # shuffled_list = df.copy()
    # random.shuffle(shuffled_list)
    for index in tqdm(range(10000)):
        if index < start_index:
            continue
        if index>=len(df):
            break
        sim.reset()
        sim.bow_head()
        time.sleep(1)
        sim.grasp('release',handSide=handSide)
        time.sleep(1)
        sim.removeObjects('all')
        objs = sim.getObjsInfo()
        desk_height = 98 # 固定桌子高度
        desk_id = 1 # random.choice(sim.desks.ID.values)
        sim.addDesk(desk_id, h=desk_height)
        ids = random.sample(list(can_list), n_objs)
        objList = [] #sim.genObjs(n=n_objs, ids=ids, h=sim.desk_height, handSide = handSide, min_distance=25)
        assert  len(df[index][0])//3==n_objs, 'data error'
        objList.append([ids[0],*df[index][0][:2],desk_height+1])
        if n_objs>1:
            objList.append([ids[1], *df[index][0][3:5],desk_height+1])
        sim.addObjects(objList)
        target_obj_index = random.randint(1,n_objs)
        if n_objs>1:
            other_obj_index = random.choice([x for x in range(1,n_objs+1) if x!=target_obj_index])
        target_origin_loc = sim.getObjsInfo()[target_obj_index]['location']
        target_obj_id = objList[target_obj_index-1][0]
        target_obj = sim.objs[sim.objs.ID == target_obj_id].Name.values[0]
        sx, sy = sim.getObservation().location.X, sim.getObservation().location.Y

        # 改变初始位置
        # x = np.random.uniform(-10, 10)
        # y = np.random.uniform(-10,10)
        # z = np.random.uniform(-2,5)
        # sim.moveHand(x,y,z,handSide=handSide, method='diff', keep_rpy=(0, 0, 0))

        ox, oy, oz = sim.getSensorsData(handSide=handSide)[0]
        offline_data['from_file'] = index
        offline_data['robot_location'] = (sx, sy, 90)
        offline_data['deskInfo'] = {'id': desk_id, 'height': sim.desk_height}
        offline_data['objList'] = objList
        offline_data['targetObjID'] = target_obj_id
        offline_data['target_obj_index'] = target_obj_index
        offline_data['initState'] = sim.getState()
        offline_data['initLoc'] = (ox-sx, oy-sy, oz)
        offline_data['handSide'] = handSide
        offline_data['event'] = args.event
        offline_data['trajectory'] = []

        last_action = (ox-sx, oy-sy, oz)
        last_img = Resize(sim.getImage())
        last_state = sim.getState()
        # do_values = []
        if args.event=='placeTargetObj':
            for action in sim.graspTargetObj(obj_id=target_obj_index,handSide=handSide):
                pass
            if not sim.checkGraspTargetObj(obj_id=target_obj_index):
                continue
        if args.event=='moveNear':
            for action in event['act'](obj1_id=target_obj_index,obj2_id=other_obj_index, handSide=handSide):
                # values = sim.bow_head()
                # do_values.append(values)
                each_frame = {}
                each_frame['img'] = last_img
                each_frame['state'] = last_state
                each_frame['action'] = action
                time.sleep(0.05)
                last_img = Resize(sim.getImage())
                last_state = sim.getState()
                each_frame['after_state'] = last_state
                offline_data['trajectory'].append(each_frame)
        else:
            for action in event['act'](obj_id=target_obj_index,handSide=handSide):
                # values = sim.bow_head()
                # do_values.append(values)
                each_frame = {}
                each_frame['img'] = last_img
                each_frame['state'] = last_state
                each_frame['action'] = action
                time.sleep(0.05)
                last_img = Resize(sim.getImage())
                last_state = sim.getState()
                each_frame['after_state'] = last_state
                offline_data['trajectory'].append(each_frame)

        if (args.event == 'moveNear' and event['check'](obj1_id=target_obj_index,obj2_id=other_obj_index)) or (args.event != 'moveNear' and event['check'](obj_id=target_obj_index)) :
            collected_num += 1
            is_success = True
            print(f'Success have collected {collected_num} datas')
            with open(output_path + f'/{collected_num:06d}.pkl', 'wb') as f:
                pickle.dump(offline_data, f)
        else:
            is_success = False
            print('fail data:', index, desk_id, target_obj_id, objList)

        im = sim.getImage()
        plt.imshow(im)
        plt.savefig(before_grasp_images_path + f"/{index:04d}_{is_success}_{target_obj}_{args.event}.png", format='png')
        # do_values = np.array(do_values)
        # np.save(log_images_path + f"/{index:04d}_{is_success}_{target_obj}.pkl", do_values)
        # 创建视频
        if len(offline_data['trajectory'])>0:
            images = [ImageClip((frame['img'] * 255).astype(np.uint8), duration=1 / 3) for frame in
                      offline_data['trajectory']]
            clip = concatenate_videoclips(images)
            clip.write_videofile(video_path + f"/{index:04d}_{is_success}_{target_obj}_{args.event}.mp4", fps=3)

        # 更新meta_data
        meta_data = {
            "collected_num": collected_num,
            "start_index": index + 1,
            "data_info": data_info,
        }
        # 将数据转换成JSON格式
        with open(meta_data_path, 'w') as f:
            json.dump(meta_data, f)