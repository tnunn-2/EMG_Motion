from typing import Any
import numpy as np
import scipy.io
import os
from pathlib import Path

GESTURE_MAP_E1 = {
    0: "Rest",
    1: "Hand open",
    2: "Hand close",
    3: "Wrist extension",
    4: "Wrist flexion",
    5: "Wrist supination",
    6: "Wrist pronation",
    7: "Wrist abduction",
    8: "Wrist adduction",
    9: "Thumb flexion",
    10: "Thumb extension",
    11: "Index flexion",
    12: "Index extension",
    13: "Middle flexion",
    14: "Middle extension",
    15: "Ring flexion",
    16: "Ring extension",
    17: "Little flexion",
    18: "Little extension"
}

GESTURE_MAP_E2 = {
    0: "Rest",
    1: "Spherical grasp",
    2: "Tip pinch",
    3: "Palmar pinch",
    4: "Lateral grasp",
    5: "Cylindrical grasp",
    6: "Hook grasp",
    7: "Platform grasp",
    8: "Tripod grasp",
    9: "Thumb–index grasp",
    10: "Thumb–middle grasp",
    11: "Thumb–ring grasp",
    12: "Thumb–little grasp",
    13: "Pointing index",
    14: "Adduction of all fingers",
    15: "Extension of all fingers"
}

GESTURE_MAP_E3 = {
    0: "Rest",
    1: "Wrist flexion + hand close",
    2: "Wrist extension + hand open",
    3: "Wrist pronation + hand close",
    4: "Wrist supination + hand open",
    5: "Wrist abduction + hand open",
    6: "Wrist adduction + hand close",
    7: "Wrist extension + finger extension",
    8: "Wrist flexion + finger flexion",
    9: "Hand open + wrist extension + thumb abduction",
    10: "Power grasp",
    11: "Key grasp",
    12: "Pinch grasp",
    13: "Pointing gesture",
    14: "OK gesture",
    15: "Thumbs up",
    16: "Finger snap",
    17: "Finger count (index to little)",
    18: "Relaxation after gesture"
}


WINDOW_SIZE = 50
OVERLAP_SIZE = 25

def ExtractMAV(signalWindow):
    """
    Calculates Mean Absolute Value (MAV) for each EMG channel.
    """
    return np.mean(np.abs(signalWindow), axis=0)

def ExtractAllFeatures(signalWindow):
    """
    Combines MAV, RMS, and VAR features into a single feature vector.
    Returns a flattened 1D array concatenating all features.
    """
    mav = ExtractMAV(signalWindow)
    rms = np.sqrt(np.mean(signalWindow ** 2, axis=0))
    var = np.var(signalWindow, axis=0)
    wl = np.sum(np.abs(np.diff(signalWindow, axis=0)), axis=0)
    zc = np.sum(np.diff(np.sign(signalWindow), axis=0) != 0, axis=0)
    return np.concatenate([mav, rms, var, wl, zc])

def SlidingWindowGenerator(emgSignals, labels, windowSize, overlapSize):
    """
    Generates windowedSignal and mostFrequentLabel for each sliding window.
    """
    stepSize = windowSize - overlapSize
    numSamples = emgSignals.shape[0]
    
    for start in range(0, numSamples - windowSize + 1, stepSize):
        end = start + windowSize
        windowedSignal = emgSignals[start:end, :]
        windowLabels = labels[start:end]
        
        # Majority vote using np.unique() and np.argmax()
        # Due to sorting of np.unique(), the lowest numerical value wins in case of tie
        uniqueLabels, counts = np.unique(windowLabels, return_counts=True)
        mostFrequentLabel = uniqueLabels[np.argmax(counts)]
        
        yield windowedSignal, mostFrequentLabel

def BuildGestureDict(emgSignals, labels, windowSize=WINDOW_SIZE, overlapSize=OVERLAP_SIZE):
    """
    Builds a dictionary mapping gesture IDs to (average MAV, list of all feature windows).
    """
    gestureDict = {}

    for windowedSignal, gestureId in SlidingWindowGenerator(emgSignals, labels, windowSize, overlapSize):
        mav = ExtractMAV(windowedSignal)
        features = ExtractAllFeatures(windowedSignal)
        
        if gestureId not in gestureDict:
            gestureDict[gestureId] = {
                "MAVs": [],
                "features": []
            }

        gestureDict[gestureId]["MAVs"].append(mav)
        gestureDict[gestureId]["features"].append(features)
    
    # Compute average MAV per gesture and format final output
    finalDict = {}
    for gestureId, data in gestureDict.items():
        avgMav = np.mean(data["MAVs"], axis=0)
        finalDict[gestureId] = (avgMav, data["features"])
    
    return finalDict

def LoadAndProcess(dataPath, windowSize=WINDOW_SIZE, overlapSize=OVERLAP_SIZE):
    """
    Loads one or more .mat files, applies sliding window segmentation,
    extracts MAV, RMS, and VAR features, and builds a gesture dictionary.
    """
    fileList = []

    if os.path.isfile(dataPath) and dataPath.lower().endswith('.mat'):
        fileList.append(dataPath)
    elif os.path.isdir(dataPath):
        print(f"Searching for .mat files in directory: {dataPath}")
        for fileName in os.listdir(dataPath):
            if fileName.lower().endswith('.mat'):
                fileList.append(os.path.join(dataPath, fileName))
    else:
        print(f"Error: Invalid path or file type provided: {dataPath}")
        return 0

    if not fileList:
        print(f"Warning: No .mat files found in {dataPath}. Check your directory structure.")
        return 0

    allFeatures = []
    allLabels = []
    combinedGestureDict = {}

    for filePath in fileList:
        print(f"Processing file: {os.path.basename(filePath)}...")
        mat = scipy.io.loadmat(filePath)

        emgSignals = mat['emg']          # (samples, channels)
        labels = mat['stimulus'].flatten()  # (samples,)

        gestureDict = BuildGestureDict(emgSignals, labels, windowSize, overlapSize)
        
        for gestureId, (avgMAV, features) in gestureDict.items():
            if gestureId not in combinedGestureDict:
                combinedGestureDict[gestureId] = (avgMAV, features)
            else:
                existingMAV, existingFeatures = combinedGestureDict[gestureId]
                combinedGestureDict[gestureId] = (
                    np.mean([existingMAV, avgMAV], axis=0),
                    existingFeatures + features
                )

        # Flatten all windows for overall dataset representation
        for gestureId, (_, features) in gestureDict.items():
            allFeatures.extend(features)
            allLabels.extend([gestureId] * len(features))

    X = np.array(allFeatures)
    Y = np.array(allLabels)

    print(f"\nSuccessfully processed {len(fileList)} file(s).")
    print(f"Final Dataset Shape: Features (X)={X.shape}, Labels (Y)={Y.shape}")
    return combinedGestureDict, X, Y

def getGestureName(exercise_num: int, gesture_id: int) -> str:
    """
    Return the descriptive name of a Ninapro DB1 gesture based on exercise and label number.
    """
    if exercise_num == 1:
        return GESTURE_MAP_E1.get(gesture_id, "Unknown")
    elif exercise_num == 2:
        return GESTURE_MAP_E2.get(gesture_id, "Unknown")
    elif exercise_num == 3:
        return GESTURE_MAP_E3.get(gesture_id, "Unknown")
    else:
        return "Unknown"

#test
if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).parent.parent 
    BASE_DATA = PROJECT_ROOT / 'data' / 'ninapro' / 'DB1'
    subjectDir = BASE_DATA / 'S1'
    if not subjectDir.is_dir():
        print(f"\nFATAL ERROR: Subject directory not found at: {subjectDir}")
        print("Please ensure your data is located in the exact structure.")
    else:
        print("\nExample 1: Loading ALL Experiment files for Subject S1")
        largeDict, XFeaturesAll, yLabelsAll = LoadAndProcess(str(subjectDir))
        print("Gestures found:", list(largeDict.keys()))
        print("Gesture 0 avg MAV shape:", largeDict[0][0].shape)
        print("Gesture 0 window count:", len(largeDict[0][1]))

        print("\nExample 2: Loading ONLY the Grasping Primitives (E2)")
        singleFile = subjectDir / 'S1_A1_E2.mat'
        singleDict, XFeaturesE2, yLabelsE2 = LoadAndProcess(str(singleFile))
        print("Gestures found:", list(singleDict.keys()))
        print("Gesture 0 avg MAV shape:", singleDict[0][0].shape)
        print("Gesture 0 window count:", len(singleDict[0][1]))