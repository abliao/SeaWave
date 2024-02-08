#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2023/09/21 11:24:46
@Author  :   alice.xiao@cloudminds.com
@File    :   0920_sim_test.py
"""
#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import sys
import time
from simUtils import *
import matplotlib.pyplot as plt
sys.path.append('./')
sys.path.append('../')

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
output_path='/data2/liangxiwen/zkd/datasets/dataGen/1_objs_3'
n_objs=1 # random.choice([2,3,4,5])
handSide='Right'
ids=12
if not os.path.exists(output_path):
    os.makedirs(output_path)
log_images_path='./log_image'
if not os.path.exists(log_images_path):
    os.makedirs(log_images_path)
grasp_images_path='./grasp_images'
if not os.path.exists(grasp_images_path):
    os.makedirs(grasp_images_path)
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
random.seed(42)
def Resize(mat):
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize((224,224))
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

from tqdm import tqdm
collected_num=1060
for epoch in range(10):
    print('Epoch:',epoch)
    offline_data=dict()
    shuffled_list = df.copy()
    random.shuffle(shuffled_list)
    for index,data in tqdm(enumerate(shuffled_list)):
        if epoch==0 and index<1087:
           continue
        data = deepcopy(data)
        sim.reset()
        desk_id=1
        sim.addDesk(desk_id=desk_id)
        objList=sim.genObjs(n=n_objs,ids=ids,handSide=handSide,h=sim.desk_height)
        obj_id = objList[0][0]
        target_obj = sim.objs[sim.objs.ID==obj_id].Name.values[0]
        sx,sy = sim.getObservation().location.X, sim.getObservation().location.Y
        offline_data['from_file']=index
        offline_data['robot_location']=(sx,sy,90)
        offline_data['deskInfo']={'id':desk_id,'height':sim.desk_height}
        offline_data['objList']=objList
        offline_data['targetObjID']=obj_id
        offline_data['initState']=sim.getState()
        offline_data['trajectory']=[]
        
        target_oringin_loc=sim.getObjsInfo()[1]['location']
        actions=[]
        images=[]
        states=[]
        dis=0
        mat = sim.getImage()
        last_img=mat
        last_state=sim.getState()
        last_action=np.array([0.,0.,0.,0.,0.,0.,0.])
        #images.append(mat)
        #states.append(sim.getState())
        have_grasp=False
        for action in sim.graspTargetObj(obj_id=1,handSide=handSide,angle=(65,68),lift_h=20,gap=0.3,return_action=True,keep_rpy=(-0,0,0)):
            actions.append(np.array(action))
            time.sleep(0.05)
            images.append(sim.getImage())
            states.append(sim.getState())
            if actions[-1][-1]==1 and (not have_grasp):
                have_grasp=True
                grasp_image=images[-1]
        for frame_id,(img,state,action) in enumerate(zip(images,states,actions)):
            each_frame={}
            if last_action[-1]!=action[-1]:
                each_frame['img']=Resize(last_img)
                each_frame['state']=last_state
                each_frame['action']=last_action.copy()
                each_frame['after_state']=states[frame_id-1]
                offline_data['trajectory'].append(each_frame)
                last_action[:6]=0
                last_state=states[frame_id-1]
                last_img=images[frame_id-1]
                dis=0
            last_action[:6]+=action[:6]
            dis+=np.linalg.norm(np.array(action[:3]))
            if dis>1 or last_action[-1]!=action[-1]:
                last_action[-1]=action[-1]
                each_frame['img']=Resize(last_img)
                each_frame['state']=last_state
                each_frame['action']=last_action.copy()
                each_frame['after_state']=state
                offline_data['trajectory'].append(each_frame)
                last_action[:6]=0
                last_state=state
                last_img=img
                dis=0
        target_now_loc=sim.getObjsInfo()[1]['location']
        if target_now_loc[2]-target_oringin_loc[2]>10:
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
        plt.imshow(grasp_image)
        plt.savefig(grasp_images_path +f"/{index:04d}_{is_success}_{target_obj}.png", format='png')