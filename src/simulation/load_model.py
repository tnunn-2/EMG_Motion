import pybullet as p
import pybullet_data
import os
import time

p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")

iiwa = p.loadURDF("kuka_iiwa/model.urdf",
                  basePosition=[0, 0, 0],
                  useFixedBase=True)

# hand_dir = os.path.join(os.path.dirname(__file__), "robot_model", "shadow_hand2")
# p.setAdditionalSearchPath(hand_dir)

# hand = p.loadURDF("shadow_hand_right.urdf",
#                   basePosition=[0, 0, 0])

EE_LINK = 6 

while p.isConnected():
    time.sleep(0.1)
