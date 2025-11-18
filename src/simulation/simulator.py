def main():
    import pybullet as p
    import pybullet_data
    import time

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")
    kuka = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    while True:
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    main()
