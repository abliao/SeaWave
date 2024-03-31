import numpy as np
import pickle
from collections import OrderedDict

def initJointsArrange():
    joints_arrange = [
        [-36,30], #全身前后（摇摆，包括腿）
        [-90,90], #躯体左右 (旋转，不是摇摆)
        [-45,45], #上半身前后（摇摆，不包括腿）
        [-45,45], #上半身左右（摇摆，不包括腿）
        [-180,180], #头旋转（扭脖子）
        [-45,36], #头前后（点头）
        [-23,23], #头左右 (摇头) 

        #左手
        [-180,36], #肩关节，整条手臂前后
        [-23,90], #肩关节，整条手臂左右
        [-90,90], #肘关节，小臂旋转
        [-120,12], #肘关节，小臂前后
        [-90,90], #腕关节，手掌旋转 [-90,90],
        [-23,23], #腕关节，手掌前后
        [-36,23], #腕关节，手掌左右

        #右手
        [-180,36], #肩关节，整条手臂前后
        [-90,23], #肩关节，整条手臂左右
        [-90,90], #肘关节，小臂旋转
        [-120,12], #肘关节，小臂前后
        [-90,90], #腕关节，手掌旋转
        [-23,23], #腕关节，手掌前后
        [-23,36], #腕关节，手掌左右
    ]
    # joints_arrange=joints_arrange/np.pi*180
    return np.array(joints_arrange,dtype=np.float32)

def get_instructions():
    f=open('instructions/database.pkl','rb')
    instructions=pickle.load(f)
    # if not isinstance(instructions,OrderedDict):
    #     instructions=OrderedDict(instructions)
    return instructions

def check_mask_id(sim):
    sim.addDesk(1)
    for id in sim.objs.ID.values:
        scene=sim.removeObjects([i for i in range(1,len(sim.registry_objs))])
        sim.genObjs(n=1,ids=int(id),h=sim.desk_height,handSide='Right')
        import matplotlib.pyplot as plt
        import numpy as np
        caremras=[GrabSim_pb2.CameraName.Head_Segment]
        ignore = np.array([0,128])
        mat=sim.getImage(caremras)
        assert (mat==ignore[0]).any() and (mat==ignore[1]).any(), 'mask may have been changed'
        unique_values = np.unique( mat.ravel())
        unique_values = np.setdiff1d(unique_values,ignore)
        assert len(unique_values)==1, 'there are other objs'
        print(id,unique_values[0])