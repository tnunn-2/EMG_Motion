import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loader import LoadAndProcess, BuildSubjectDict, WindowData
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from DANN import DANN, train_dann, few_shot_on_dann
import torch
import os
from show_results import plot_loso_results, plot_single_subject_results

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def BuildRestData(gestureDict):
    """
    Creates xRest and yRest given a gestureDict
    """
    xRest = []
    yRest = []
    for gestureId, (_, features) in gestureDict.items():
        for feature in features:
            xRest.append(feature)
            if gestureId == 0:
                yRest.append(0)
            else:
                yRest.append(1)
    
    xRest = np.array(xRest)
    yRest = np.array(yRest)

    print(f"Rest Detector Dataset Created:")
    print(f"  XRest Shape = {xRest.shape}")
    print(f"  yRest Distribution: Rest = {(yRest == 0).sum()}, Active = {(yRest == 1).sum()}")

    return xRest, yRest

def BuildActiveData(gestureDict):
    xActive = []
    yActive = []
    for gestureId, (_, features) in gestureDict.items():
        if gestureId != 0:
            xActive.extend(features)
            yActive.extend([gestureId]*len(features))
    xActive = np.array(xActive)
    yActive = np.array(yActive)

    return xActive, yActive

def train(gestureDict):
    x, y = BuildRestData(gestureDict)
    xTrainRest, xTestRest, yTrainRest, yTestRest = train_test_split(x, y, test_size=0.2)
    
    restScaler = StandardScaler()
    xTrainScaledRest = restScaler.fit_transform(xTrainRest)
    xTestScaledRest = restScaler.transform(xTestRest)
    restClf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    restClf.fit(xTrainScaledRest, yTrainRest)


    xActive, yActive = BuildActiveData(gestureDict)
    xTrainActive, xTestActive, yTrainActive, yTestActive = train_test_split(xActive, yActive, test_size = 0.2)
    gestureClf, scaler = tuneSVC(xTrainActive, yTrainActive)
    xTestScaled = scaler.transform(xTestActive)

    yPredGesture = gestureClf.predict(xTestScaled)
    yPredRest = restClf.predict(xTestScaledRest)

    print("Rest Detector Accuracy: {:.4f}".format(accuracy_score(yTestRest, yPredRest)))
    print("Gesture Classifier (SVC) Accuracy: {:.4f}".format(accuracy_score(yTestActive, yPredGesture)))

    # --- CONFUSION MATRIX & AUC METRICS ---
    print("\n=== Rest Detector Metrics ===")
    cm_rest = confusion_matrix(yTestRest, yPredRest)
    print("Confusion Matrix (Rest Detector):\n", cm_rest)
    
    # For binary rest classifier
    if len(np.unique(yTestRest)) == 2:
        yScoreRest = restClf.predict_proba(xTestScaledRest)[:, 1]
        auc_rest = roc_auc_score(yTestRest, yScoreRest)
        print(f"ROC AUC (Rest Detector): {auc_rest:.4f}")
    
    print("\n=== Gesture Classifier Metrics ===")
    cm_gesture = confusion_matrix(yTestActive, yPredGesture)
    print("Confusion Matrix (Gesture Classifier):\n", cm_gesture)
    
    # For multi-class AUC (one-vs-rest)
    try:
        yScoreGesture = gestureClf.decision_function(xTestScaled)
        auc_gesture = roc_auc_score(yTestActive, yScoreGesture, multi_class='ovr')
        print(f"Mean ROC AUC (Gesture Classifier): {auc_gesture:.4f}")
    except Exception:
        print("Skipping gesture AUC (decision_function not supported for this model).")

    # --- SAMPLE PREDICTIONS ---
    print("\nSample Gesture Predictions (True → Predicted):")
    for i in range(10):
        print(f"  {int(yTestActive[i]):>2} → {int(yPredGesture[i])}")


    return restClf, gestureClf, restScaler, scaler
    

def tuneSVC(xActive, yActive):
    """
    Tunes hyperparamters for the gesture classifier (SVC with RBF)
    """
    print("Tuning SVC Hyperparamters")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True))
    ])

    paramGrid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 0.01, 0.001]
    }

    grid = GridSearchCV(pipe, paramGrid, cv=3, n_jobs=1, verbose=3)
    grid.fit(xActive, yActive)

    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

    bestModel = grid.best_estimator_
    scaler = bestModel.named_steps['scaler']
    gestureClf = bestModel.named_steps['svc']

    return gestureClf, scaler

def losoEvaluate(
    allSubjectDict,
    subjects_to_test=None,
    few_shot_K=None,
    knn_neighbors=3,
    do_prototype=True,
    do_knn=True,
    do_dann=True,
    dann_epochs=20,
    dann_batch=128,
    device="cpu",
):
    """
    LOSO evaluation with:
    - baseline SVC classifier
    - optional subject subset
    - few-shot calibration (prototype + kNN)
    - optional DANN domain adaptation (neural net)
    """

    subjects = sorted(list(allSubjectDict.keys()))

    # subjects_to_test controls only WHICH subjects we evaluate
    if subjects_to_test is None:
        test_subjects = subjects
    else:
        test_subjects = subjects_to_test


    results = {
        "baseline": {},
        "fewshot_prototype": {},
        "fewshot_knn": {},
        "dann": {},
        "finetune_dann": {}
    }

    print(f"\nRunning LOSO on subjects: {subjects}")
    if few_shot_K is not None:
        print(f"Few-shot calibration enabled: K={few_shot_K}\n")

    for test_subj in test_subjects:
        print(f"\nLeaving Out Subject {test_subj}")
        train_subjects = [s for s in subjects if s != test_subj]

        X_train = np.vstack([allSubjectDict[s]['features'] for s in train_subjects])
        y_train = np.hstack([allSubjectDict[s]['labels'] for s in train_subjects])

        X_test = allSubjectDict[test_subj]['features']
        y_test = allSubjectDict[test_subj]['labels']

        scaler = StandardScaler().fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        train_gesture_dict = defaultdict(lambda: (None, []))
        for s in train_subjects:
            for gid, (_, feats) in allSubjectDict[s]['gestureDict'].items():
                train_gesture_dict[gid][1].extend(feats)

        restClf, gestureClf, restScaler, svc_scaler = train(train_gesture_dict)
        y_pred_baseline = gestureClf.predict(svc_scaler.transform(X_test))
        acc_base = accuracy_score(y_test, y_pred_baseline)
        results["baseline"][test_subj] = acc_base
        print(f"[Baseline LOSO] Accuracy: {acc_base:.4f}")

        if few_shot_K is None:
            continue

        class_idxs = defaultdict(list)
        for idx, g in enumerate(y_test):
            class_idxs[g].append(idx)

        calib_idxs = []
        for g, idxs in class_idxs.items():
            if len(idxs) <= few_shot_K:
                calib_idxs.extend(idxs)
            else:
                calib_idxs.extend(list(np.random.choice(idxs, few_shot_K, replace=False)))

        calib_idxs = sorted(calib_idxs)

        X_calib = X_test[calib_idxs]
        y_calib = y_test[calib_idxs]

        eval_idxs = [i for i in range(len(y_test)) if i not in calib_idxs]
        X_eval = X_test[eval_idxs]
        y_eval = y_test[eval_idxs]

        scaler_fs = StandardScaler().fit(np.vstack([X_train, X_calib]))
        X_train_fs = scaler_fs.transform(X_train)
        X_calib_fs = scaler_fs.transform(X_calib)
        X_eval_fs = scaler_fs.transform(X_eval)

        if do_prototype:
            prototypes = {}
            classes = np.unique(y_train)

            for c in classes:
                prototypes[c] = X_train_fs[y_train == c].mean(axis=0)

            for c in classes:
                mask = (y_calib == c)
                if np.any(mask):
                    prototypes[c] = 0.5 * prototypes[c] + 0.5 * X_calib_fs[mask].mean(axis=0)

            y_pred_proto = [
                min(prototypes.keys(),
                    key=lambda c: np.linalg.norm(x - prototypes[c]))
                for x in X_eval_fs
            ]

            acc_proto = accuracy_score(y_eval, y_pred_proto)
            results["fewshot_prototype"][test_subj] = acc_proto
            

        if do_knn:
            knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
            knn.fit(np.vstack([X_train_fs, X_calib_fs]),
                    np.hstack([y_train, y_calib]))

            y_pred_knn = knn.predict(X_eval_fs)
            acc_knn = accuracy_score(y_eval, y_pred_knn)
            results["fewshot_knn"][test_subj] = acc_knn
            
        
        if do_dann:
            results, encoderPath, headPath = doDANN(train_subjects, 
                    X_train, y_train, X_test, y_test, test_subj,
                    dann_batch, dann_epochs, device,
                    results)


        if do_prototype:
            print(f"[Few-shot Prototype] Accuracy: {acc_proto:.4f}")
        if do_knn:
            print(f"[Few-shot kNN-{knn_neighbors}] Accuracy: {acc_knn:.4f}")
    return results, encoderPath, headPath, restClf, restScaler, gestureClf, scaler

def doDANN(train_subjects, X_train, y_train, X_test, y_test, test_subj,
           dann_batch, dann_epochs, device, results):

    X_train, y_train = WindowData(X_train, y_train, window=20, stride=5)
    X_test,  y_test  = WindowData(X_test,  y_test,  window=20, stride=5)

    #Reshape from (N, T, C) → (N, C, T)
    X_train = np.transpose(X_train, (0, 2, 1))   # (N, 70, 20)
    X_test  = np.transpose(X_test,  (0, 2, 1))   # (N, 70, 20)

    print(f"\n --- Training DANN on LOSO split (source={train_subjects}, target={test_subj}) ---")

    src_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    tgt_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.zeros(len(X_test), dtype=torch.long)
    )

    src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=dann_batch, shuffle=True)
    tgt_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size=dann_batch, shuffle=True)

    model = DANN(
        input_channels=X_train.shape[1],
        window_size=X_train.shape[2],
        hidden_dim=128,
        num_classes=len(np.unique(y_train))
    ).to(device)

    train_dann(model, src_loader, tgt_loader, num_epochs=dann_epochs, device=device)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        class_out, _ = model(xt)
        dann_preds = torch.argmax(class_out, dim=1).cpu().numpy()

    acc_dann = np.mean(dann_preds == y_test)
    results["dann"][test_subj] = acc_dann
    print(f"[DANN LOSO] Accuracy: {acc_dann:.4f}")

    fs_results = few_shot_on_dann(
        model,
        X_train, y_train,
        X_test, y_test,
        few_shot_K=5,
        n_neighbors=3,
        alpha=0.5,
        finetune=True,
        ft_epochs=20,
        ft_lr=1e-3,
        device=device
    )

    print("DANN-latent few-shot (before finetune): proto = {:.4f}, knn = {:.4f}".format(
        fs_results['proto_before_acc'], fs_results['knn_before_acc']
    ))
    if fs_results.get('head') is not None:
        print("After finetune: proto = {:.4f}, knn = {:.4f}".format(
            fs_results.get('proto_after_acc'), fs_results.get('knn_after_acc')
        ))

    results["finetune_dann"][test_subj] = fs_results

    save_dir = "results"
    savedEncoderPath = f"{save_dir}/encoder_subject{test_subj}_epochs{dann_epochs}.pt"
    savedHeadPath = f"{save_dir}/head_subject{test_subj}_epochs{dann_epochs}.pt"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.encoder.state_dict(), savedEncoderPath)
    if fs_results.get("head") is not None:
        torch.save(fs_results["head"].state_dict(), savedHeadPath)

    return results, savedEncoderPath, savedHeadPath

def predictGesture(emg_frame, restClf, gestureClf, restScaler, scaler):
    """
    Takes a single EMG feature vector and returns:
        0  --> rest
        1,2,3,... --> Ninapro gesture integer
    """
    X = np.array(emg_frame).reshape(1, -1)

    X_rest = restScaler.transform(X)
    rest_pred = restClf.predict(X_rest)[0]

    if rest_pred == 0:
        return 0

    X_active = scaler.transform(X)
    gesture_id = int(gestureClf.predict(X_active)[0])

    return gesture_id

#Test
if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).parent.parent 
    BASE_DATA = PROJECT_ROOT / 'data' / 'ninapro' / 'DB1'
    subjectDir = BASE_DATA / 'E1'
    singleFile = subjectDir / 'S1_A1_E1.mat'

    # print("\nLoading EMG data from: ", subjectDir)
    # singleDict, xE2, yE2 = LoadAndProcess(str(subjectDir))
    # print("Gestures found:", list(singleDict.keys()))


    # print("\nTraining model on data...")
    # restClf, gestureClf, restScaler, scaler = train(singleDict)

    subjectList = [1, 2, 3]
    testSubj = 2
    eNum = 2
    print("\nLoading EMG data from: ", subjectDir)
    singleDict = BuildSubjectDict(subjectList, eNum)
    print("Subjects found:", list(singleDict.keys()))


    print("\nTraining model on data and evaluating via LOSO...")
    results, _, _, _, _, _, _ = losoEvaluate(singleDict, subjects_to_test=[testSubj], few_shot_K=5)
    plot_single_subject_results(results, subj=testSubj)