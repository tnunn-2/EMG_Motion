import time
import pybullet as p
from load_model import connect_gui, load_kuka, step
from gesture_detector import predict_gesture
from gesture_detector import restClf, gestureClf, restScaler, scaler


GESTURE_TASK_MAP = {
    0: "rest",

    1: "move_up",
    2: "move_down",
    3: "extend_forward",
    4: "retract",
    5: "rotate_left",
    6: "rotate_right",

    13: "open_gripper",
    14: "close_gripper",

    20: "go_home_pose",
    21: "shake_left",
    22: "shake_right",

    "default": "rest"
}

IIWA_TASK_MAP = {
    "rest":          [0, 0, 0, 0, 0, 0, 0],
    "move_up":       [0.3, 0.2, 0, 0, 0, 0, 0],
    "move_down":     [-0.3, -0.2, 0, 0, 0, 0, 0],
    "extend_forward":[0, 0, 0.4, 0.3, 0, 0, 0],
    "retract":       [0, 0, -0.4, -0.3, 0, 0, 0],
    "rotate_left":   [0, 0, 0, 0.6, 0, 0, 0],
    "rotate_right":  [0, 0, 0, -0.6, 0, 0, 0],

    "go_home_pose":  [0, -1.0, 1.5, 0.5, 0, -1.0, 0],
}


def map_gesture_to_task(gesture_id):
    return GESTURE_TASK_MAP.get(gesture_id, GESTURE_TASK_MAP["default"])

def map_task_to_joints(task):
    motion = IIWA_TASK_MAP.get(task)
    if isinstance(motion, list):
        return motion
    else:
        print("Non-joint task:", task)
        return None

def get_gesture():
    return int(time.time()) % 6
    # return predict_gesture(emg_frame, restClf, gestureClf, restScaler, scaler)


STEP = 0.05  # movement step for IK

x, y, z = 0.4, 0.0, 0.6
targetOri = p.getQuaternionFromEuler([0, 0, 0])


connect_gui()
iiwa, EE = load_kuka()



while p.isConnected():

    gesture_id = get_gesture()

    print("Detected Gesture ID:", gesture_id)


    task = map_gesture_to_task(gesture_id)
    print("Mapped Task:", task)


    joint_targets = map_task_to_joints(task)


    if joint_targets is not None:
        for j in range(7):
            p.setJointMotorControl2(
                iiwa,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_targets[j],
                force=200
            )

    else:
        if task == "move_up":
            z += STEP
        elif task == "move_down":
            z -= STEP
        elif task == "extend_forward":
            x += STEP
        elif task == "retract":
            x -= STEP
        elif task == "rotate_left":
            y += STEP
        elif task == "rotate_right":
            y -= STEP

        jointPoses = p.calculateInverseKinematics(
            iiwa,
            EE,
            [x, y, z],
            targetOri
        )

        for j in range(7):
            p.setJointMotorControl2(
                iiwa,
                j,
                p.POSITION_CONTROL,
                targetPosition=jointPoses[j],
                force=200
            )

    step()
    time.sleep(1/240)
