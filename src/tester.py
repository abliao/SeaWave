import hydra
from omegaconf import DictConfig
import re
from pathlib import Path
import pickle
import random
import logging
from copy import deepcopy
import time
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, ImageClip, concatenate_videoclips

from google.protobuf import message
import grpc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from Env import SimEnv, GrabSim_pb2_grpc, GrabSim_pb2, initJointsArrange
from Env.simUtils import *
from Env.gen_data import name_type,gen_objs


actuatorRanges=np.array([[-30.00006675720215, 31.65018653869629],
 [-110.00215911865234, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-159.9984588623047, 129.99838256835938],
 [-15.000033378601074, 150.00035095214844],
 [-5.729577541351318, 64.74422454833984],
 [-30.00006675720215, 30.00006675720215],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-45.00010299682617, 58.49898910522461],
 [-39.999900817871094, 39.999900817871094],
 [-90.00020599365234, 90.00020599365234],
 [-45.00010299682617, 45.00010299682617],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-110.00215911865234, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-129.99838256835938, 159.9984588623047],
 [-150.00035095214844, 15.000033378601074],
 [-5.729577541351318, 64.74422454833984],
 [-30.00006675720215, 30.00006675720215],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234]])


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
    mat = mat.resize((224,224)) 
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def grasp(sim,agent,log,robot_location,objList,device='cuda',history_len=1,handSide='Right',control='joint'):
    robot_location=np.array(robot_location)
    instr=log['instruction']

    max_steps=80
    max_collision=0
    obs=Resize(sim.getImage())
    img=torch.Tensor(obs)
    img=img.reshape(-1,1,*img.shape).permute(0,1,4,2,3).to(device)
    imgs=torch.repeat_interleave(img, history_len, dim=1)
    target_oringin_loc=sim.getObjsInfo()[1]['location']
    state_targe_loc = objList[0][1:4]
    sensors=sim.getState()['sensors']
    # state = np.concatenate([sensors[3]['data'],state_targe_loc])
    state = np.array(sensors[3]['data'])
    state[:3]-=robot_location
    for sensor in sensors[4:]:
        if 'right' in sensor['name']:
            state = np.concatenate([state,np.array(sensor['data'])-robot_location])
    state[:]/=10
    state=torch.Tensor(state).to(device).unsqueeze(0).unsqueeze(0)
    states=torch.repeat_interleave(state, history_len, dim=1)
    log['imgs']=[sim.getImage()]
    joints = np.array(sim.getActuators())
    for _ in range(max_steps):
        sim.bow_head()
        time.sleep(0.03)
        obs=Resize(sim.getImage())
        img=torch.Tensor(obs)
        img=img.reshape(-1,1,*img.shape).permute(0,1,4,2,3).to(device)
        sensors=sim.getState()['sensors']
        # state = np.concatenate([sensors[3]['data'],state_targe_loc])
        state = np.array(sensors[3]['data'])
        state[:3]-=robot_location
        for sensor in sensors[4:]:
            if 'right' in sensor['name']:
                state = np.concatenate([state,np.array(sensor['data'])-robot_location])
        state[:]/=10
        state=torch.Tensor(state).to(device).unsqueeze(0).unsqueeze(0)
        
        if history_len==1:
            imgs = img
            states = state
        else:
            imgs = torch.cat([imgs[:,-history_len+1:],img],dim=1)
            states = torch.cat([states[:,-history_len+1:],state],dim=1)
        assert imgs.shape[1]==history_len, f"length of input sequence is error, needed {history_len} not {imgs.shape[1]}"
        # print('states',states)
        batch={}
        batch['observations']=img
        batch['states']=state
        batch['instr']=[instr]
        predict=agent.act(batch)
        predict=predict[0].cpu().detach().numpy()
        last_action=predict
        if control=='ee':
            if sim.grasp_state[handSide]==0:
                msg=sim.moveHand(x=last_action[0],y=last_action[1],z=last_action[2],keep_rpy=(0,0,0),method='diff',gap=0.1,handSide=handSide)
            else:
                msg=sim.moveHand(x=last_action[0],y=last_action[1],z=last_action[2],method='diff',gap=0.1,handSide=handSide)
        else:
            now_joints = np.array(sim.getActuators())
            joint_ids = [-12,-11,-6,-5]
            joints[joint_ids] = now_joints[joint_ids]
            # print('last_action',last_action)
            last_action[:4] = last_action[:4]*(actuatorRanges[joint_ids,1]-actuatorRanges[joint_ids,0])/50
            # print('decode_last_action', last_action)
            joints[joint_ids] += last_action[:4]
            # print('last_action',last_action)
            sim.changeJoints(joints)
        
        if_grasp = sigmoid(last_action[-1])>0.5

        if if_grasp and sim.grasp_state[handSide]==0:
            sim.grasp(angle=(65,68))
            # time.sleep(3)
            print(f'to grasp, grasp_state={sim.grasp_state[handSide]}')
            log['grasp_img'] = sim.getImage()
        elif not if_grasp and sim.grasp_state[handSide]==1:
            sim.release()
        
        log['track'].append(last_action.copy())
        log['imgs'].append(sim.getImage())
        target_now_loc=sim.getObjsInfo()[1]['location']
        if target_now_loc[2]-target_oringin_loc[2]>10:
            log['info']='success'
            break

        if _==max_steps-1:
            log['info']='time_exceed'
            break
    print('target',sim.getObjsInfo()[1]['name'])
    print('target_oringin_loc',target_oringin_loc)
    print('target_now_loc',target_now_loc)
    return log

def Tester(agent,cfg,episode_dir):
    levels = cfg.datasets.eval.levels
    client=cfg.env.client
    history_len = cfg.datasets.history_len
    action_nums=cfg.env.num_actions
    bins = cfg.env.bins
    mode = cfg.env.mode
    control = cfg.env.control
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
        sim.EnableEndPointCtrl(True)
        sim.reset()
        # sim.changeWrist(0,0,-40)
        # sim.moveHand(-2.0311660766601562, -46.86720657399303, 116.66156768798828,method='relatively')
        if control=='joint':
            sim.EnableEndPointCtrl(False)
        else:
            sim.EnableEndPointCtrl(True)
        desk_id=1
        sim.addDesk(desk_id=desk_id)
        can_list=[12,14,16,17,18]
        obj_id = random.choice(can_list)
        other_obj_ids = random.choices([x for x in sim.objs.ID.values if x!=obj_id],k=n_objs-1)
        ids = [obj_id]+other_obj_ids
        objList=sim.genObjs(n=n_objs,ids=ids,handSide=handSide,h=sim.desk_height)
        
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
        log=grasp(sim,agent,log,robot_location=robot_location,objList=objList,device=device,history_len=history_len,control=control)
        
        images = [ImageClip(frame.astype(np.uint8), duration=1/6) for frame in log['imgs']]
        # 创建视频
        clip = concatenate_videoclips(images)

        

        del log['imgs']
        logs.append(log)

        if log['info']=='success':
            success+=1

        total_num+=1
        logging.info(f'num: {total_num}, success rate:{success/total_num*100:.2f}%)')
        print('Instruction: ',instr)
        time.sleep(1)
        if log['info'] in ['success','collision','time_exceed']: 
            print('targetObj:',log['targetObj'])
            print(f"done at {len(log['track'])} steps")
            print(log['detail'])
            
            # if index==0:
            im=sim.getImage()
            plt.imshow(im)
            plt.savefig(episode_dir / f"{index:04d}_{log['info']}_{log['targetObj']}.png", format='png')
            if 'grasp_img' in log.keys():
                im=log['grasp_img']
                plt.imshow(im)
                plt.savefig(episode_dir / f"{index:04d}_grasp_{log['info']}_{log['targetObj']}.png", format='png')
            with open(episode_dir /'trajectory.pkl','wb') as f:
                pickle.dump(logs,f)
            clip.write_videofile(str(episode_dir / f"{index:04d}_grasp_{log['info']}_{log['targetObj']}.mp4"), fps=6)

    # sim.setLightIntensity(0.5)