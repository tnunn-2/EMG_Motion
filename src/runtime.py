# runtime.py
import argparse
from pathlib import Path
import time
import os
import joblib
import pickle
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p

# your project imports (adjust import paths if needed)
from gesture_detector import losoEvaluate
from show_results import plot_single_subject_results
from simulation.load_model import connect_gui, load_kuka, step  # existing functions you already have
from DANN import DANN, get_latents
from simulation.emg_to_iiwa import run_robot_realtime  # DANN.py new CORAL/CNN implementation
from data_loader import BuildSubjectDict, ExtractAllFeatures, SlidingWindowGenerator, WindowData
import scipy.io as scipy


# ---------------------------
# Default mapping and params
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
    "default": "rest"
}

IIWA_TASK_MAP = {
    "rest":          [0, 0, 0, 0, 0, 0, 0],
    "go_home_pose":  [0, -1.0, 1.5, 0.5, 0, -1.0, 0],
}

# Real-time parameters
WINDOW_SIZE = 20        # frames per window (must match model.window_size)
OVERLAP_SIZE = 15
CHANNELS = 70           # your feature count
BUFFER_MAXLEN = WINDOW_SIZE
SMOOTH_K = 5            # history length for majority vote
STEP = 0.02             # Cartesian step (m) per prediction for translational gestures
SLEEP = 1/240.0         # step rate for PyBullet

# ---------------------------
# Utilities
# ---------------------------
def load_scalers(scaler_path, rest_scaler_path=None):
    scaler = joblib.load(scaler_path)
    rest_scaler = joblib.load(rest_scaler_path) if rest_scaler_path else None
    return scaler, rest_scaler

def load_prototypes(proto_path):
    # expected: dict {class_int: prototype_vector}
    with open(proto_path, "rb") as f:
        return pickle.load(f)

def load_knn(knn_path):
    with open(knn_path, "rb") as f:
        return pickle.load(f)

def nearest_prototype(prototypes, feat):
    """
    prototypes: dict class->vector (1D numpy)
    feat: (D,) numpy
    returns nearest class (int)
    """
    keys = list(prototypes.keys())
    vals = np.vstack([prototypes[k] for k in keys])
    dists = np.linalg.norm(vals - feat.reshape(1, -1), axis=1)
    arg = np.argmin(dists)
    return keys[arg]

def majority_vote(history_deque, new_pred):
    history_deque.append(new_pred)
    # return majority
    items, counts = np.unique(np.array(history_deque), return_counts=True)
    return items[np.argmax(counts)]

# ---------------------------
# Windowing / preprocessing
# ---------------------------
class SlidingWindowBuffer:
    """
    Holds the most recent raw-frame feature vectors for windowing.
    Each incoming sample must be a (C,) numpy array (C = CHANNELS).
    Produces windows of shape (C, WINDOW_SIZE) when full.
    """
    def __init__(self, window_size=WINDOW_SIZE, channels=CHANNELS):
        self.window_size = window_size
        self.channels = channels
        self.buf = deque(maxlen=window_size)

    def add_sample(self, sample):
        """
        sample: shape (C,) or list length C
        returns: None or window (C, window_size) when buffer full
        """
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.channels:
            raise ValueError(f"Incoming sample must be 1D length {self.channels}")
        self.buf.append(arr)
        if len(self.buf) == self.window_size:
            # stack -> (T, C), transpose -> (C, T)
            window = np.stack(self.buf, axis=0).T.copy()
            return window
        else:
            return None


# ---------------------------
# Inference functions
# ---------------------------
def predict_from_model_window(model, window, device="cpu", mode="logits", prototypes=None, knn=None):
    """
    window: numpy (C, T)
    mode: "logits" | "proto" | "knn"
    returns: predicted class int, latent vector (1, D) numpy
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, T)
        logits, latent = model(xb)      # logits (1, num_classes), latent (1, D)
        latent_np = latent.cpu().numpy().squeeze(0)  # (D,)
        if mode == "logits":
            pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
            return pred, latent_np
        elif mode == "proto":
            if prototypes is None:
                raise ValueError("prototypes required for proto mode")
            pred = nearest_prototype(prototypes, latent_np)
            return int(pred), latent_np
        elif mode == "knn":
            if knn is None:
                raise ValueError("knn required for knn mode")
            pred = int(knn.predict(latent_np.reshape(1, -1))[0])
            return pred, latent_np
        else:
            raise ValueError("Unknown mode")

# ---------------------------
# Robot control mapping
# ---------------------------
def map_gesture_to_task(gesture_id):
    return GESTURE_TASK_MAP.get(gesture_id, GESTURE_TASK_MAP["default"])

def apply_task_to_robot(iiwa, EE, task, x, y, z, targetOri):
    """
    If task maps to fixed joint pose in IIWA_TASK_MAP, use that.
    Otherwise apply delta to (x,y,z) and solve IK.
    Returns updated (x,y,z)
    """
    # check joint tasks
    if task in IIWA_TASK_MAP and not (task == "rest"):
        joint_targets = IIWA_TASK_MAP[task]
        for j in range(7):
            p.setJointMotorControl2(iiwa, j, p.POSITION_CONTROL, targetPosition=joint_targets[j], force=200)
        return x, y, z

    # else treat as Cartesian increment for movement tasks
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
    # solve IK
    jointPoses = p.calculateInverseKinematics(iiwa, EE, [x, y, z], targetOri)
    for j in range(7):
        p.setJointMotorControl2(iiwa, j, p.POSITION_CONTROL, targetPosition=jointPoses[j], force=200)
    return x, y, z


def load_ninapro_file(f):
    mat = scipy.loadmat(f)
    emg = mat["emg"]                # (T,10)
    labels = mat["stimulus"].flatten()

    return preprocess_realtime_emg(emg, labels)


def preprocess_realtime_emg(emg_raw, labels_raw,
                            windowSize=WINDOW_SIZE,
                            overlapSize=OVERLAP_SIZE,
                            temporal_window=20,
                            temporal_stride=5):
    """
    Full preprocessing pipeline that reproduces EXACTLY
    what training used:
    - SlidingWindowGenerator on raw EMG
    - ExtractAllFeatures() → 70-dim feature vectors
    - WindowData() → (20, 70) temporal windows
    - Transpose to (20, 70)
    
    Returns:
        X_stream: (N, 20, 70)
        y_stream: (N,)
    """

    feature_list = []
    label_list = []

    # Step 1: raw EMG → sliding windows → extract 70-dim features
    for windowedSignal, label in SlidingWindowGenerator(emg_raw, labels_raw,
                                                       windowSize, overlapSize):
        feat = ExtractAllFeatures(windowedSignal)  # shape (70,)
        feature_list.append(feat)
        label_list.append(label)

    features = np.array(feature_list)       # (T_feat, 70)
    labels = np.array(label_list)

    # Step 2: Apply WindowData over features (temporal structure)
    X_tw, y_tw = WindowData(features, labels,
                            window=temporal_window,
                            stride=temporal_stride)
    # X_tw: (N, 70, 20)

    # Step 3: Transpose to match encoder input
    X_tw = np.transpose(X_tw, (0, 2, 1))    # (N, 20, 70)

    return X_tw, y_tw



def main(mode="train"):
    """
    mode = "train"     trains LOSO, saves encoder/head/scalers/classifiers
    mode = "realtime"  loads saved models and runs realtime control
    mode = "both"      trains LOSO, saves params, and performs realtime   
    """

    subjectList = [1, 2, 3]
    testSubj = 2
    eNum = 2

    PROJECT_ROOT = Path(__file__).parent.parent 
    BASE_DATA = PROJECT_ROOT / 'data' / 'ninapro' / 'DB1'
    new_patient_files = [BASE_DATA / f'S{testSubj}' / f'S{testSubj}_A1_E{eNum}.mat']

    SAVE_DIR = "saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder_path = f"{SAVE_DIR}/encoder.pt"
    head_path    = f"{SAVE_DIR}/head.pt"
    restclf_path = f"{SAVE_DIR}/restClf.pkl"
    restscaler_path = f"{SAVE_DIR}/restScaler.pkl"
    gestureclf_path  = f"{SAVE_DIR}/gestureClf.pkl"
    scaler_path = f"{SAVE_DIR}/scaler.pkl"


    if mode == "train":
        print("[1] Building training dataset...")
        subject_dict = BuildSubjectDict(subjectList, eNum)

        print("[2] LOSO training...")
        results, encPath, headPath, restClf, restScaler, gestureClf, scaler = losoEvaluate(
            subject_dict,
            subjects_to_test=[testSubj],
            few_shot_K=5,
            knn_neighbors=3,
            do_prototype=True,
            do_knn=True,
            do_dann=True,
            dann_epochs=50,
            dann_batch=128,
            device="cpu"
        )

        # Visualization
        plot_single_subject_results(results, subj=testSubj)

        print("[Saving runtime artifacts...]")
        torch.save(torch.load(encPath), encoder_path)
        torch.save(torch.load(headPath), head_path)

        joblib.dump(restClf,     restclf_path)
        joblib.dump(restScaler,  restscaler_path)
        joblib.dump(gestureClf,  gestureclf_path)
        joblib.dump(scaler,      scaler_path)

        print("[Training complete]")
        print("Saved:")
        print(" encoder:", encoder_path)
        print(" head:", head_path)
        print(" restClf:", restclf_path)
        print(" restScaler:", restscaler_path)
        print(" gestureClf:", gestureclf_path)
        print(" scaler:", scaler_path)
        return


    if mode == "realtime" or "both":
        print("\n[Realtime Mode] Loading saved model artifacts...")

        required_files = [
            encoder_path, head_path, restclf_path,
            restscaler_path, gestureclf_path, scaler_path
        ]
        for fpath in required_files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"\nERROR: Missing {fpath}\n"
                    "You must run main(mode='train') first."
                )

        restClf = joblib.load(restclf_path)
        restScaler = joblib.load(restscaler_path)
        gestureClf = joblib.load(gestureclf_path)
        scaler = joblib.load(scaler_path)

        ckpt = torch.load(encoder_path)
        print("Loaded encoder checkpoint.")

        in_ch = ckpt["conv.0.weight"].shape[1]       # = 20
        fc_in = ckpt["fc.weight"].shape[1]          # = 4480

        # last conv has 64 output channels (from checkpoint)
        last_ch = ckpt["conv.6.weight"].shape[0]    # = 64

        T_final = fc_in // last_ch                  # = 4480 / 64 = 70
        # window_size = T_final + 10                  # add conv shrinking = 80
        window_size = T_final

        print(f"Inferred: input_channels={in_ch}, window_size={window_size}")

        encoder = DANN(
            input_channels=in_ch,
            window_size=window_size,
            hidden_dim=128,
            num_classes=12
        ).encoder

        encoder.load_state_dict(ckpt)
        encoder.eval()
        print("Encoder loaded.")

        # Load head safely
        ckpt_head = torch.load(head_path)
        num_classes_ckpt = ckpt_head['weight'].size(0)  # infer from checkpoint
        head = torch.nn.Linear(128, num_classes_ckpt)
        head.load_state_dict(ckpt_head)
        head.eval()
        print(f"Head loaded with {num_classes_ckpt} classes.")



        print("[Loading new patient file for realtime simulation...]")

        # Corrected realtime Ninapro loader
        X_new_list = []
        y_new_list = []

        for f in new_patient_files:
            Xf, yf = load_ninapro_file(f)   # no need for window_size argument anymore
            X_new_list.append(Xf)          # each → (Nf, 20, 70)
            y_new_list.append(yf)

        # concatenate 3D arrays properly
        X_new = np.concatenate(X_new_list, axis=0)
        y_new = np.concatenate(y_new_list, axis=0)

        # shuffle
        perm = np.random.permutation(len(y_new))
        X_new = X_new[perm]
        y_new = y_new[perm]

        print("[Starting REALTIME robot control + evaluation...]")
        print("X_new:", X_new.shape)
        print("Example frame:", X_new[0].shape)


        run_robot_realtime(
            encoder=encoder,
            head=head,
            X_stream=X_new,
            y_stream=y_new,
            restClf=restClf,
            restScaler=restScaler,
            gestureClf=gestureClf,
            scaler=scaler
        )
        return


    raise ValueError("mode must be 'train' or 'realtime'")



def diagnose_and_load_encoder(encoder_path, hidden_dim=128, num_classes=12):
    print(">> Inspecting checkpoint:", encoder_path)
    ckpt = torch.load(encoder_path, map_location="cpu")
    print("Checkpoint keys and shapes:")
    for k,v in ckpt.items():
        try:
            print(" ", k, getattr(v, "shape", type(v)))
        except Exception:
            print(" ", k, type(v))

    # Strategy A: If checkpoint contains 'fc.weight' we can compute window_size
    if 'fc.weight' in ckpt:
        fc_shape = ckpt['fc.weight'].shape  # (out, in)
        fc_in = fc_shape[1]
        # infer channel count at conv output: check last conv out-channels
        # find last conv weight like conv.*.weight where weight.ndim==3 and last has shape [C_out, C_in, k]
        conv_keys = [k for k in ckpt.keys() if k.startswith('conv') and k.endswith('.weight') and getattr(ckpt[k], 'ndim', 0)==3]
        if not conv_keys:
            raise RuntimeError("No conv weight keys found in checkpoint")
        # pick last conv key by sorted order
        conv_keys_sorted = sorted(conv_keys, key=lambda s: int(s.split('.')[1]))
        last_conv = conv_keys_sorted[-1]
        last_conv_shape = ckpt[last_conv].shape  # (C_out, C_in, k)
        last_conv_out = last_conv_shape[0]
        inferred_window = fc_in // last_conv_out
        print(f"Inferred: fc_in={fc_in}, last_conv_out={last_conv_out} => window_size_inferred={inferred_window}")
        # instantiate model with input_channels from first conv weight
        first_conv_key = conv_keys_sorted[0]
        first_conv_shape = ckpt[first_conv_key].shape
        input_channels = first_conv_shape[1]
        print(f"Inferred input_channels from checkpoint: {input_channels}")
        encoder = DANN(input_channels=input_channels, window_size=inferred_window, hidden_dim=hidden_dim, num_classes=num_classes).encoder
        print("Instantiated encoder with:", input_channels, inferred_window, hidden_dim)
    else:
        # If fc.weight absent, maybe checkpoint is whole model or different structure — try to inspect conv.0.weight
        if 'conv.0.weight' in ckpt:
            first = ckpt['conv.0.weight'].shape
            print("Found conv.0.weight shape:", first)
            input_channels = first[1]
            # can't infer window_size without fc; fallback to provided default
            raise RuntimeError("Checkpoint missing fc.weight; cannot infer window_size automatically.")
        else:
            raise RuntimeError("Checkpoint missing expected conv/fc keys; print above to inspect.")

    # Print model keys
    print("\nModel keys:")
    for k, v in encoder.state_dict().items():
        print(" ", k, v.shape)

    # Try to load exactly (strict=True) and show helpful diagnostics on failure
    try:
        encoder.load_state_dict(ckpt)
        print("\nLoaded checkpoint into encoder with strict=True ✅")
    except Exception as e:
        print("\nStrict load failed:", str(e))
        # show which keys mismatch / missing
        try:
            missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
            print("Loaded with strict=False. Missing keys:", missing)
            print("Unexpected keys in checkpoint:", unexpected)
            # warn if shapes mismatched
            # check each key present in both: compare shapes
            for k in ckpt:
                if k in encoder.state_dict():
                    if ckpt[k].shape != encoder.state_dict()[k].shape:
                        print(f"Shape mismatch key={k}: ckpt {ckpt[k].shape} vs model {encoder.state_dict()[k].shape}")
        except Exception as e2:
            print("Also failed with strict=False:", e2)
            raise

    # Test forward pass with dummy input to ensure shapes OK
    try:
        C = input_channels
        T = inferred_window
        x = torch.randn(2, C, T)
        with torch.no_grad():
            out = encoder(x)
        print("Forward pass OK: encoder output shape:", out.shape)
    except Exception as e:
        print("Forward pass failed:", e)
        raise

    return encoder, ckpt

# usage (replace encoder_path with your variable)
# encoder, ckpt = diagnose_and_load_encoder(encoder_path)



if __name__ == "__main__":
    # main(mode="realtime",
    # encoder_path="results/encodersubject_2_epochs20.pt",
    # head_path="results/head_subject2_epochs20.pt")

    # main(mode="train")
    main(mode = "realtime")

    # main(mode="both")

    # diagnose_and_load_encoder(encoder_path="results/encodersubject_2_epochs20.pt")