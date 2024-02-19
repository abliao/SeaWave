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

print('数据生成脚本启动')

host = '127.0.0.1:30008'
scene_num = 1
map_id = 2
server = SimServer(host, scene_num=scene_num, map_id=map_id)

sim = Sim(host, scene_id=0)

import pickle

with open('Imitation_data/RLexpert/0718_single_merge_data_new.pkl', 'rb') as f:
    df = pickle.load(f)

import os

output_path = '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_4'
data_info="使用中指判断位置,逼近过程中保持摄像头不变,转动关节时其余关节使用旧状态,增加第二段轨迹长度"
meta_data_path = output_path + os.sep + 'meta_data.json'
n_objs = 1
handSide = 'Right'
can_list = [12, 14, 16, 17, 18]
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

random.seed(42)


def Resize(mat):
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize((224, 224))
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat / 255.0
    return mat


from tqdm import tqdm

collected_num = meta_data['collected_num']
start_index = meta_data['start_index']
for epoch in range(1):
    print('Epoch:', epoch)
    offline_data = dict()
    shuffled_list = df.copy()
    random.shuffle(shuffled_list)
    for index, data in tqdm(enumerate(shuffled_list)):
        if index < start_index:
            continue
        data = deepcopy(data)
        sim.reset()
        sim.bow_head()
        time.sleep(1)
        sim.grasp('release',handSide=handSide)
        time.sleep(1)
        sim.removeObjects('all')
        objs = sim.getObjsInfo()
        scene = sim.removeObjects([0])
        loc = data['obj_loc']
        desk_height = 98 # 固定桌子高度
        desk_id = 1
        sim.addDesk(desk_id, h=desk_height)
        obj_id = random.choice(can_list)
        other_obj_ids = random.choices([x for x in sim.objs.ID.values if x != obj_id], k=n_objs - 1)
        ids = [obj_id] + other_obj_ids
        objList = sim.genObjs(n=n_objs, ids=ids, h=sim.desk_height, handSide = handSide)
        target_origin_loc = sim.getObjsInfo()[1]['location']
        obj_id = objList[0][0]
        target_obj = sim.objs[sim.objs.ID == obj_id].Name.values[0]
        XX, YY, _ = data['robot_location']
        sx, sy = sim.getObservation().location.X, sim.getObservation().location.Y

        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10,10)
        z = np.random.uniform(-2,5)
        sim.moveHand(x,y,z,handSide=handSide, method='diff', keep_rpy=(0, 0, 0))

        ox, oy, oz = sim.getSensorsData(handSide=handSide)[0]
        offline_data['from_file'] = index
        offline_data['robot_location'] = (sx, sy, 90)
        offline_data['deskInfo'] = {'id': desk_id, 'height': loc[2]}
        offline_data['objList'] = objList
        offline_data['targetObjID'] = obj_id
        offline_data['initState'] = sim.getState()
        offline_data['initLoc'] = (ox-sx, oy-sy, oz)
        offline_data['handSide'] = handSide
        offline_data['trajectory'] = []
        last_action = (ox-sx, oy-sy, oz)
        last_img = Resize(sim.getImage())
        last_state = sim.getState()
        do_values = []
        for action in sim.closeTargetObj(obj_id=1,handSide=handSide,return_action=True):
            values = sim.bow_head()
            do_values.append(values)
            each_frame = {}
            each_frame['img'] = last_img
            each_frame['state'] = last_state
            each_frame['action'] = (*action, 0)
            time.sleep(0.05)
            last_img = Resize(sim.getImage())
            last_state = sim.getState()
            each_frame['after_state'] = last_state
            offline_data['trajectory'].append(each_frame)

        loc1 = sim.findObj(name=target_obj)['location']
        loc2 = sim.getSensorsData(handSide=handSide)[0]

        # 抓取
        each_frame = {}
        mat = sim.getImage()
        mat = Resize(mat)
        each_frame['img'] = mat
        each_frame['state'] = sim.getState()
        before_grasp_img = sim.getImage()
        sim.grasp(handSide=handSide)
        grasp_img = sim.getImage()
        each_frame['action'] = (0, 0, 0, 0, 0, 0, 1)
        each_frame['after_state'] = sim.getState()
        offline_data['trajectory'].append(each_frame)

        last_img = sim.getImage()
        last_state = sim.getState()
        for action in sim.moveHandReturnAction(0, 0, 10, gap=1, method='diff',handSide=handSide):
            # values = sim.bow_head()
            # do_values.append(values)
            each_frame = {}
            each_frame['img'] = Resize(last_img)
            each_frame['state'] = last_state
            if handSide=='Right':
                each_frame['action'] = (0, 0, 0, 0, 0, 1, 1)
            else:
                each_frame['action'] = (0, 0, 1, 0, 0, 0, 1)
            state = sim.getState()
            each_frame['after_state'] = state
            offline_data['trajectory'].append(each_frame)
            last_img = sim.getImage()
            last_state = state

        target_now_loc = sim.getObjsInfo()[1]['location']
        if target_now_loc[2] - target_origin_loc[2] > 10:
            collected_num += 1
            is_success = True
            print(f'Success have collected {collected_num} datas')
            with open(output_path + f'/{collected_num:06d}.pkl', 'wb') as f:
                pickle.dump(offline_data, f)
        else:
            is_success = False
            print('fail data:', index, desk_id, obj_id, objList)

        im = sim.getImage()
        plt.imshow(im)
        plt.savefig(log_images_path + f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
        plt.imshow(grasp_img)
        plt.savefig(grasp_images_path + f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
        plt.imshow(before_grasp_img)
        plt.savefig(before_grasp_images_path + f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
        # do_values = np.array(do_values)
        # np.save(log_images_path + f"/{index:04d}_{is_success}_{target_obj}.pkl", do_values)
        # 创建视频
        images = [ImageClip((frame['img'] * 255).astype(np.uint8), duration=1 / 3) for frame in
                  offline_data['trajectory']]
        clip = concatenate_videoclips(images)
        clip.write_videofile(video_path + f"/{index:04d}_{is_success}_{target_obj}.mp4", fps=3)

        # 更新meta_data
        meta_data = {
            "collected_num": collected_num,
            "start_index": index + 1,
            "data_info": data_info,
        }
        # 将数据转换成JSON格式
        with open(meta_data_path, 'w') as f:
            json.dump(meta_data, f)