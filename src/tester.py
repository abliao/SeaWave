import hydra
from omegaconf import DictConfig
import re
from pathlib import Path
import pickle
import random
from copy import deepcopy
import time
from tqdm import tqdm

from google.protobuf import message
import grpc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from Env import SimEnv, GrabSim_pb2_grpc, GrabSim_pb2, initJointsArrange
from Env.simUtils import *
from Env.gen_data import name_type,gen_objs




def read_data(path):
    import re
    # f=open('RLexpert/0816_two_obj_data.txt')
    f=open(path)
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
    return df

def action_untokenization(env, action,bins,joints_arrange):
    # action=action.argmax(axis=-1)
    
    joints=action*(joints_arrange[-7:,1]-joints_arrange[-7:,0])/50
    return joints

def genObjwithLists(sim_client,sceneID,objList):
    for x,y,z,yaw,type in objList:
        obj_list = [GrabSim_pb2.ObjectList.Object(x=x, y=y, yaw=yaw, z=z, type=type)]
        # obj_list = [GrabSim_pb2.ObjectList.Object(X=ginger_loc[0] + x_rand, Y=ginger_loc[1] + y_rand, Yaw=yaw_rand, Z=h, type=type_rand)]
        scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))

def get_image(sim_client,sceneID):
    caremras=[GrabSim_pb2.CameraName.Head_Color]
    action = GrabSim_pb2.CameraList(sceneID=sceneID, cameras=caremras)
    im = sim_client.Capture(action).images[0]
    mat = np.frombuffer(im.data,dtype=im.dtype).reshape((im.height, im.width, im.channels))
    return mat

def get_depth(sim_client,sceneID):
    caremras=[GrabSim_pb2.CameraName.Head_Depth]
    action = GrabSim_pb2.CameraList(sceneID=sceneID, cameras=caremras)
    im = sim_client.Capture(action).images[0]
    mat = np.frombuffer(im.data,dtype=im.dtype).reshape((im.height, im.width, im.channels))
    t=100 #150
    mat = 1.0 * mat
    mat[mat>t]=t
    return mat
        
datas=[]

def is_element_in_string(element_list, target_string):
    for element in element_list:
        if element in target_string:
            return True
    return False

from PIL import Image
def Resize(mat):
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize((224,244)) 
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def grasp(sim,agent,log,robot_location,device='cuda'):
    robot_location=np.array(robot_location)
    instr=log['instruction']

    max_steps=50
    max_collision=0
    target_oringin_loc=sim.getObjsInfo()[1]['location']
    for _ in range(max_steps):
        obs=Resize(sim.getImage())
        img=torch.Tensor(obs)
        img=img.reshape(-1,1,*img.shape).permute(0,1,4,2,3).to(device)
        
        sensors=sim.getState()['sensors']
        state = np.array(sensors[3]['data'])
        wrist_loc = state.copy()
        wrist_loc[:2]-=robot_location[:2]
        state[:3]-=robot_location
        state[:3]/=10
        state=torch.Tensor(state).to(device).unsqueeze(0)
        
        batch={}
        batch['observations']=img
        batch['states']=state
        batch['instr']=[instr]
        import time
        predict=agent.act(batch)
        predict=predict[0].cpu().detach().numpy()
        last_action=predict
        msg=sim.moveHand(x=last_action[0],y=last_action[1],z=last_action[2],keep_rpy=(0,0,0),method='diff',gap=0.1)
        if sigmoid(last_action[-1])>0.5:
            sim.grasp()
        log['track'].append(last_action.copy())
        
        target_now_loc=sim.getObjsInfo()[1]['location']
        if target_now_loc[2]-target_oringin_loc[2]>10:
            log['info']='success'
            break

        if _==max_steps-1:
            log['info']='time exceed'
            break
        
    return log

def Tester_level0(agent,cfg,episode_dir):
    levels = cfg.datasets.eval.levels
    client=cfg.env.client
    action_nums=cfg.env.num_actions
    bins = cfg.env.bins
    mode = cfg.env.mode
    max_steps = cfg.env.max_steps
    device = cfg.common.device
    agent.load(**cfg.initialization,device=device)
    agent.to(device)
    agent.eval()
                     
    scene_num = 1
    map_id = 2
    server = SimServer(client,scene_num = scene_num, map_id = map_id)
    sim=Sim(client,scene_id=0)
    
    with open('/data2/liangxiwen/RM-PRT/IL/RLexpert/0718_single_merge_data_new.pkl','rb') as f:
        df=pickle.load(f)

    success=0
    rule_success=0
    rule_num=0
    total_num=0

    
    with open(cfg.datasets.test.instructions_path,'rb') as f:
        instructions=pickle.load(f)

    logs=[]
    for index,data in tqdm(enumerate(random.choices(df,k=400))):
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
        # obj_id = random.choice(sim.objs.ID.values)
        # target_obj_id = obj_id
        loc4gen=loc[:]
        loc4gen[2]+=1
        # objList = [[obj_id,*loc4gen,0,0,0]]
        # scene=sim.addObjects([[obj_id,*loc4gen,0,0,0]])
        n_objs=1 # random.choice([2,3,4,5])
        # objList = sim.genObjs(n=n_objs,target_loc=loc4gen[:2],h=loc[2])
        objList = sim.genObjs(n=n_objs,h=loc[2])
        target_obj_id = objList[0][0]
        XX,YY, ZZ = data['robot_location']
        assert ZZ==0,'z!=0'
        ZZ=90
        sx,sy = sim.getObservation().location.X, sim.getObservation().location.Y
        for frame in data['traj'][:1]:
            x,y,z=frame['details'][-1]['location']
            x = (x-XX)
            y = (y-YY)
            z+=5
            x+=1
            sim.moveHand(x=x,y=y,z=z,method='relatively')
        
        log={}
        log['objs']=objList
        log['deskInfo']={'desk_id':desk_id,'height':loc[2]}
        log['initLoc']=(x,y,z)
        log['detail']=''
        log['track']=[]
        log['targetObjID']=target_obj_id
        log['targetObj']=sim.objs[sim.objs.ID==log['targetObjID']].Name.values[0]

        instr = 'pick a ' + log['targetObj']
        log['instruction']=instr
        
        robot_location = sim.getObservation().location.X, sim.getObservation().location.Y, 90
        log=grasp(sim,agent,log,log['initLoc'],robot_location=robot_location,device=device)
        
        logs.append(log)

        if log['info']=='success':
            success+=1

        total_num+=1
        print(f'num: {total_num}, success rate:{success/total_num*100:.2f}%)')
        print('Instruction: ',instr)
        time.sleep(1)
        if log['info'] in ['success','collision','time exceed']: 
            print('targetObj:',log['targetObj'])
            print(f"done at {len(log['track'])} steps")
            print(log['detail'])
            
            # if index==0:
            im=sim.getImage()
            plt.imshow(im)
            plt.savefig(episode_dir / f"{index:04d}_{log['info']}_{log['targetObj']}.png", format='png')
            with open(episode_dir /'trajectory.pkl','wb') as f:
                pickle.dump(logs,f)

def Tester(agent,cfg,episode_dir):
    levels = cfg.datasets.eval.levels
    client=cfg.env.client
    action_nums=cfg.env.num_actions
    bins = cfg.env.bins
    mode = cfg.env.mode
    max_steps = cfg.env.max_steps
    device = cfg.common.device
    agent.load(**cfg.initialization,device=device)
    agent.to(device)
    agent.eval()
                     
    scene_num = 1
    map_id = 2
    server = SimServer(client,scene_num = scene_num, map_id = map_id)
    sim=Sim(client,scene_id=0)

    success=0
    rule_success=0
    rule_num=0
    total_num=0

    
    with open(cfg.datasets.test.instructions_path,'rb') as f:
        instructions=pickle.load(f)

    logs=[]
    n_objs=1
    handSide='Right'
    for index in tqdm(range(100)):
        sim.reset()
        desk_id=1
        sim.addDesk(desk_id=desk_id)
        objList=sim.genObjs(n=n_objs,handSide=handSide,h=sim.desk_height)
        obj_id = objList[0][0]
        target_obj_id = obj_id
        targetObj = sim.objs[sim.objs.ID==obj_id].Name.values[0]
        

        log={}
        log['objs']=objList
        log['deskInfo']={'desk_id':desk_id,'height':sim.desk_height}
        log['detail']=''
        log['track']=[]

        log['targetObjID']=target_obj_id
        log['targetObj']=targetObj

        instr = 'pick a ' + targetObj
        log['instruction']=instr
        
        sx,sy = sim.getObservation().location.X, sim.getObservation().location.Y
        robot_location = (sx,sy,90)
        log=grasp(sim,agent,log,robot_location=robot_location,device=device)
        
        logs.append(log)

        if log['info']=='success':
            success+=1

        total_num+=1
        print(f'num: {total_num}, success rate:{success/total_num*100:.2f}%)')
        print('Instruction: ',instr)
        time.sleep(1)
        if log['info'] in ['success','collision','time exceed']: 
            print('targetObj:',log['targetObj'])
            print(f"done at {len(log['track'])} steps")
            print(log['detail'])
            
            # if index==0:
            im=sim.getImage()
            plt.imshow(im)
            plt.savefig(episode_dir / f"{index:04d}_{log['info']}_{log['targetObj']}.png", format='png')
            with open(episode_dir /'trajectory.pkl','wb') as f:
                pickle.dump(logs,f)
    # sim.setLightIntensity(0.5)