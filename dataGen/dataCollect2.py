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

host = '127.0.0.1:30006'
scene_num = 1
map_id = 2
server = SimServer(host,scene_num = scene_num, map_id = map_id)

sim=Sim(host,scene_id=0)

import pickle
with open('Imitation_data/RLexpert/0718_single_merge_data_new.pkl','rb') as f:
    df = pickle.load(f)

import os
output_path='/data2/liangxiwen/RM-PRT/3_objs2'
if not os.path.exists(output_path):
    os.makedirs(output_path)
log_images_path='./log_image2'
if not os.path.exists(log_images_path):
    os.makedirs(log_images_path)
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
random.seed(42)
def Resize(mat):
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize((224,244))
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

from tqdm import tqdm
collected_num=0
for epoch in range(10):
    print('Epoch:',epoch)
    offline_data=dict()
    shuffled_list = df.copy()
    random.shuffle(shuffled_list)
    for index,data in tqdm(enumerate(shuffled_list)):
        # if epoch==0 and index<=307:
        #     continue
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
        desk_id = random.choice(sim.desks.ID.values)
        sim.addDesk(desk_id,h=loc[2])
        obj_id = random.choice(sim.objs.ID.values)
        loc4gen=loc[:]
        loc4gen[2]+=1
        # scene=sim.addObjects([[obj_id,*loc4gen,0,0,0]])
        n_objs=3 # random.choice([2,3,4,5])
        objList = sim.genObjs(n=n_objs,target_loc=loc4gen[:2],h=loc[2])
        obj_id = objList[0][0]
        target_obj = sim.objs[sim.objs.ID==obj_id].Name.values[0]
        XX,YY, _ = data['robot_location']
        sx,sy = sim.getObservation().location.X, sim.getObservation().location.Y
        for frame in data['traj'][:1]:
            x,y,z=frame['details'][-1]['location']
            x = (x-XX)
            y = (y-YY)
            z+=5
            x+=1
            sim.moveHand(x=x,y=y,z=z,method='relatively')
        offline_data['from_file']=index
        offline_data['robot_location']=(sx,sy,90)
        offline_data['deskInfo']={'id':desk_id,'height':loc[2]}
        offline_data['objList']=objList
        offline_data['targetObjID']=obj_id
        offline_data['initState']=sim.getState()
        offline_data['initLoc']=(x,y,z)
        offline_data['trajectory']=[]
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
            x+=1
            sim.moveHand(x=x,y=y,z=z,method='relatively')
            sim.changeWrist(10,0,0)
            each_frame['action']=(x,y,z)
            each_frame['after_state']=sim.getState()
            offline_data['trajectory'].append(each_frame)

        tx,ty,tz=sim.findObj(id=1)['location']
        each_frame = {}
        time.sleep(0.05)
        mat = sim.getImage()
        mat = Resize(mat)
        time.sleep(0.05)
        each_frame['img']=mat
        each_frame['state']=sim.getState()
        sim.moveHand(x=0,y=0,z=-1,method='diff',gap=0.1)
        each_frame['action']=(x,y,z-1)
        each_frame['after_state']=sim.getState()
        offline_data['trajectory'].append(each_frame)


        ox,oy,oz = sim.getSensorsData('Right')[0]
        ex,ey,ez = tx+12.806174701878561-0,ty-7.65568545918587+4,tz-1.2643331596129599
        k = int(max(np.abs([ex-ox,ey-oy,ez-oz]))/1)+1
        for nx,ny,nz in np.linspace([ox,oy,oz],[ex,ey,ez],k+1)[1:]:
            each_frame = {}
            time.sleep(0.05)
            mat = sim.getImage()
            mat = Resize(mat)
            time.sleep(0.05)
            each_frame['img']=mat
            each_frame['state']=sim.getState()
            sim.moveHand(nx,ny,nz,method='absolute')
            each_frame['action']=(nx-sx,ny-sy,nz)
            each_frame['after_state']=sim.getState()
            offline_data['trajectory'].append(each_frame)
        # sim.grasp(angle=30)
        loc1=sim.findObj(name=target_obj)['location']
        loc2=sim.getSensorsData('Right')[0]
        if (loc2[0]-loc1[0])**2+(loc2[1]-loc1[1])**2<20**2 and abs(loc2[2]-loc1[2])<3 and loc2[0]-loc1[0]>8 and loc2[1]-loc1[1]<-3:
            collected_num+=1
            is_success=True
            print(f'Success have collected {collected_num} datas')
            with open(output_path+f'/{collected_num:06d}.pkl','wb') as f:
                pickle.dump(offline_data,f)
        else:
            is_success=False
            print('Failed',(loc2[0]-loc1[0])**2+(loc2[1]-loc1[1])**2)
            print('fail data:',index,desk_id,obj_id,loc)
        im=sim.getImage()
        plt.imshow(im)
        plt.savefig(log_images_path +f"/{index:04d}_{is_success}_{target_obj}.png", format='png')
