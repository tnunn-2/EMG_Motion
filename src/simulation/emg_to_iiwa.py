import time
import pybullet as p
import torch
from collections import deque
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from .load_model import connect_gui, load_kuka, step

# ---------------------------
# Gesture & Task Mapping
# ---------------------------
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

STEP = 0.05  # Cartesian step for non-joint tasks

# ---------------------------
# Utilities
# ---------------------------
def map_gesture_to_task(gesture_id):
    return GESTURE_TASK_MAP.get(gesture_id, GESTURE_TASK_MAP["default"])

def map_task_to_joints(task):
    motion = IIWA_TASK_MAP.get(task)
    if isinstance(motion, list):
        return motion
    else:
        print("Non-joint task:", task)
        return None

def predict_with_dann(frame, encoder, head, scaler):
    """
    frame: 1x70 feature vector (already extracted from Ninapro)
    Returns: gesture_id (integer)
    """
    X = scaler.transform(frame.reshape(1, -1))
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (1,70,1)
    h = encoder(X_t)
    logits = head(h)
    g = int(torch.argmax(logits, dim=1).cpu().item())
    return g

# ---------------------------
# Real-time robot loop
# ---------------------------
def run_robot_realtime(encoder, head, X_stream, y_stream, restClf=None, restScaler=None, gestureClf=None, scaler=None, shuffle=True):
    """
    Reworked realtime loop: use encoder+head directly on (20,70) frames.
    Old classical ML rest/gesture detectors are ignored because training
    used (20,70) inputs and the scalers expect 70-dim vectors.
    """
    if shuffle:
        perm = np.random.permutation(len(y_stream))
        X_stream = X_stream[perm]
        y_stream = y_stream[perm]

    # Connect PyBullet
    connect_gui()
    iiwa, EE = load_kuka()
    targetOri = p.getQuaternionFromEuler([0, 0, 0])

    # Initialize robot position
    x, y, z = 0.4, 0.0, 0.6

    # Metrics accumulators
    pred_list = []
    true_list = []
    vote_buf = deque(maxlen=5)
    total_count = 0
    correct_count = 0

    # Determine device for encoder/head
    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        # encoder probably not a torch.nn.Module? default to cpu
        device = torch.device("cpu")

    encoder.eval()
    head.eval()

    try:
        for i, frame in enumerate(X_stream):
            # frame expected shape: (20, 70)
            # Convert to tensor shaped (1, 20, 70) and to correct device
            tensor_frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)

            # Optional lightweight rest detector (commented)
            # If you really want a rest detector, derive one from features:
            # energy = torch.norm(tensor_frame)   # or np.linalg.norm(frame)
            # if energy < REST_ENERGY_THRESHOLD: 
            #     gesture_id = 0
            # else:

            # Inference: encoder -> head
            with torch.no_grad():
                # If your encoder expects (batch, channels, window) this is correct.
                # If your model expects another ordering, adjust permute() here.
                latent = encoder(tensor_frame)               # e.g. (1, latent_dim)
                logits = head(latent)                        # (1, num_classes)
                gesture_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

            # Smoothing / majority vote
            vote_buf.append(gesture_id)
            smoothed_gesture = max(set(vote_buf), key=vote_buf.count)

            # Map gesture → task
            task = map_gesture_to_task(smoothed_gesture)
            joint_targets = map_task_to_joints(task)

            # Apply joint motion
            if joint_targets is not None:
                for j in range(7):
                    p.setJointMotorControl2(
                        iiwa, j, p.POSITION_CONTROL,
                        targetPosition=joint_targets[j],
                        force=200
                    )
            else:
                # Non-joint Cartesian updates
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

                jointPoses = p.calculateInverseKinematics(iiwa, EE, [x, y, z], targetOri)
                for j in range(7):
                    p.setJointMotorControl2(
                        iiwa, j, p.POSITION_CONTROL,
                        targetPosition=jointPoses[j],
                        force=200
                    )

            step()
            time.sleep(1/240)

            # --- Performance accumulation ---
            true_label = y_stream[i]
            pred_list.append(smoothed_gesture)
            true_list.append(true_label)
            total_count += 1
            if smoothed_gesture == true_label:
                correct_count += 1

            if total_count % 100 == 0:
                print(f"[Realtime Eval] Frames {total_count}, running accuracy: {correct_count/total_count:.3f}", flush=True)

    except KeyboardInterrupt:
        print("Realtime loop interrupted by user.", flush=True)
    finally:
        p.disconnect()
        print("PyBullet disconnected safely.", flush=True)

        # --- Final metrics ---
        if len(true_list) == 0:
            print("=== Real-time evaluation finished ===")
            print("Total frames: 0, no predictions were made.")
            return

        overall_acc = accuracy_score(true_list, pred_list)
        cm = confusion_matrix(true_list, pred_list)
        print(f"=== Real-time evaluation finished ===")
        print(f"Total frames: {total_count}, Overall accuracy: {overall_acc:.3f}")
        print("Confusion matrix:\n", cm)


