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

host = '127.0.0.1:30008'
scene_num = 1
map_id = 2
server = SimServer(host,scene_num = scene_num, map_id = map_id)

sim=Sim(host,scene_id=0)

import pickle
with open('Imitation_data/RLexpert/0718_single_merge_data_new.pkl','rb') as f:
    df = pickle.load(f)

import os
output_path='/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_5'
meta_data_path = output_path+os.sep+'meta_data.json'
n_objs=1
can_list = [12,14,16,17,18]
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(meta_data_path):
    meta_data = {
    "collected_num": 0,
    "start_index": 0,
    }
    # 将数据转换成JSON格式
    with open(meta_data_path,'w') as f:
        json.dump(meta_data,f)
else:
    with open(meta_data_path,'rb') as f:
        meta_data = json.load(f)
    
log_images_path='./log_images'
if not os.path.exists(log_images_path):
    os.makedirs(log_images_path)
grasp_images_path='./grasp_images'
if not os.path.exists(grasp_images_path):
    os.makedirs(grasp_images_path)
before_grasp_images_path='./before_grasp_images'
if not os.path.exists(before_grasp_images_path):
    os.makedirs(before_grasp_images_path)
video_path='./log_videos'
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
    mat = mat.resize((224,224))
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

from tqdm import tqdm

collected_num = meta_data['collected_num']
start_index = meta_data['start_index']
for epoch in range(1):
    print('Epoch:',epoch)
    offline_data=dict()
    shuffled_list = df.copy()
    random.shuffle(shuffled_list)
    for index,data in tqdm(enumerate(shuffled_list)):
        if index<start_index:
            continue
        data = deepcopy(data)
        sim.reset()
        sim.bow_head()
        time.sleep(1)
        sim.grasp('release')
        time.sleep(1)
        sim.changeWrist(0,0,-40)
        sim.removeObjects('all')
        objs=sim.getObjsInfo()
        scene=sim.removeObjects([0])
        loc=data['obj_loc']
        loc[2]-=9.425
        desk_id = 1
        sim.addDesk(desk_id,h=loc[2])
        obj_id = random.choice(can_list)
        other_obj_ids = random.choices([x for x in sim.objs.ID.values if x!=obj_id],k=n_objs-1)
        loc4gen=loc[:]
        loc4gen[2]+=1
        loc4gen[0]-=5
        # scene=sim.addObjects([[obj_id,*loc4gen,0,0,0]])
        
        ids = [obj_id]+other_obj_ids
        objList = sim.genObjs(n=n_objs,ids=ids,target_loc=loc4gen[:2],h=loc[2])
        target_origin_loc=sim.getObjsInfo()[1]['location']
        obj_id = objList[0][0]
        target_obj = sim.objs[sim.objs.ID==obj_id].Name.values[0]
        XX,YY, _ = data['robot_location']
        sx,sy = sim.getObservation().location.X, sim.getObservation().location.Y
        
        for frame in data['traj'][:1]:
            x,y,z=frame['details'][-1]['location']
            x = (x-XX)
            y = (y-YY)
            z+=5
            # x+=3
            sim.moveHand(x=x,y=y,z=z,method='relatively',keep_rpy=(0,0,0))
            sim.bow_head()
        offline_data['from_file']=index
        offline_data['robot_location']=(sx,sy,90)
        offline_data['deskInfo']={'id':desk_id,'height':loc[2]}
        offline_data['objList']=objList
        offline_data['targetObjID']=obj_id
        offline_data['initState']=sim.getState()
        offline_data['initLoc']=(x,y,z)
        offline_data['trajectory']=[]
        last_action = (x,y,z)
        for frame in data['traj'][1:]:
            each_frame = {}
            time.sleep(0.03)
            mat = sim.getImage()
            mat = Resize(mat)
            time.sleep(0.03)
            each_frame['img']=mat
            each_frame['state']=sim.getState()

            x,y,z=frame['details'][-1]['location']
            x = (x-XX)
            y = (y-YY)
            z+=5
            # x+=3
            sim.moveHand(x=x,y=y,z=z,method='relatively',keep_rpy=(0,0,0))
            sim.bow_head()
            # sim.changeWrist(10,0,0)
            each_frame['action']=(x-last_action[0],y-last_action[1],z-last_action[2],0,0,0,0)
            last_action = (x,y,z)
            each_frame['after_state']=sim.getState()
            offline_data['trajectory'].append(each_frame)

        
        each_frame = {}
        time.sleep(0.05)
        mat = sim.getImage()
        mat = Resize(mat)
        time.sleep(0.05)
        each_frame['img']=mat
        each_frame['state']=sim.getState()
        sim.moveHand(x=0,y=0,z=-1,method='diff',gap=0.1,keep_rpy=(0,0,0))
        sim.bow_head()
        each_frame['action']=(0,0,-1,0,0,0,0)
        each_frame['after_state']=sim.getState()
        offline_data['trajectory'].append(each_frame)

        tx,ty,tz=sim.findObj(id=1)['location']
        ox,oy,oz = sim.getSensorsData('Right')[0]
        ex,ey,ez = tx+10,ty-1.5,tz-2
        k = int(max(np.abs([ex-ox,ey-oy,ez-oz]))/1)+1
        last_action = (ox-sx,oy-sy,oz)
        for nx,ny,nz in np.linspace([ox,oy,oz],[ex,ey,ez],k+1)[1:]:
            each_frame = {}
            time.sleep(0.05)
            mat = sim.getImage()
            mat = Resize(mat)
            time.sleep(0.05)
            each_frame['img']=mat
            each_frame['state']=sim.getState()
            sim.moveHand(nx,ny,nz,method='absolute',keep_rpy=(0,0,0))
            sim.bow_head()
            each_frame['action']=(nx-sx-last_action[0],ny-sy-last_action[1],nz-last_action[2],0,0,0,0)
            last_action = (nx-sx,ny-sy,nz)
            each_frame['after_state']=sim.getState()
            offline_data['trajectory'].append(each_frame)
        # sim.grasp(angle=30)
        loc1=sim.findObj(name=target_obj)['location']
        loc2=sim.getSensorsData('Right')[0]

        # 抓取
        each_frame = {}
        mat = sim.getImage()
        mat = Resize(mat)
        each_frame['img']=mat
        each_frame['state']=sim.getState()
        before_grasp_img = sim.getImage()
        sim.grasp()
        grasp_img = sim.getImage()
        each_frame['action']=(0,0,0,0,0,0,1)
        each_frame['after_state']=sim.getState()
        offline_data['trajectory'].append(each_frame)

        last_img = sim.getImage()
        last_state=sim.getState()
        for action in sim.moveHandReturnAction(0,0,15,gap=1,method='diff'):
            sim.bow_head()
            each_frame={}
            each_frame['img']=Resize(last_img)
            each_frame['state']=last_state
            each_frame['action'] = (0,0,1,0,0,0,1)
            state = sim.getState()
            each_frame['after_state']=state
            offline_data['trajectory'].append(each_frame)
            last_img = sim.getImage()
            last_state = state
        
        target_now_loc=sim.getObjsInfo()[1]['location']
        if target_now_loc[2]-target_origin_loc[2]>10:
            collected_num+=1
            is_success=True
            print(f'Success have collected {collected_num} datas')
            with open(output_path+f'/{collected_num:06d}.pkl','wb') as f:
                pickle.dump(offline_data,f)
        else:
            is_success=False
            print('fail data:',index,desk_id,obj_id,objList)

        im=sim.getImage()
        plt.imshow(im)
        plt.savefig(log_images_path +f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
        plt.imshow(grasp_img)
        plt.savefig(grasp_images_path +f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
        plt.imshow(before_grasp_img)
        plt.savefig(before_grasp_images_path +f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
        
        # 创建视频
        images = [ImageClip((frame['img']*255).astype(np.uint8), duration=1/3) for frame in offline_data['trajectory']]
        clip = concatenate_videoclips(images)
        clip.write_videofile(video_path +f"/{index:04d}_{is_success}_{target_obj}.mp4", fps=3)

        # 更新meta_data
        meta_data = {
            "collected_num": collected_num,
            "start_index": index+1,
            }
        # 将数据转换成JSON格式
        with open(meta_data_path,'w') as f:
            json.dump(meta_data,f)