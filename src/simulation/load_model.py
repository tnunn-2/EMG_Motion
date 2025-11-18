import pybullet as p
import pybullet_data
import os

def load_kuka():
    """
    Loads the KUKA iiwa arm and returns:
    - robot ID
    - end-effector link index (6)
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    iiwa = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True
    )

    ee_link = 6 
    return iiwa, ee_link


def connect_gui():
    """Connect to PyBullet in GUI mode and set gravity."""
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)


def step():
    p.stepSimulation()
