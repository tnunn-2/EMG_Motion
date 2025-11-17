import pybullet as p
import pybullet_data
import os
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    hand_folder = r"C:\Projects\EMG_Motion\src\simulation\robot_model\shadow_hand"
    urdf_path = os.path.join(hand_folder, "shadow_hand.urdf")

    print("Loading:", urdf_path)

    hand_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.2],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION
    )

    p.setGravity(0, 0, -9.8)

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    main()
