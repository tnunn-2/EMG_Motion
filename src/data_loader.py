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

def ExtractMavFeatures(signalWindow):
    """
    Calculates Mean Absolute Value (MAV) for each EMG channel.
    This feature is used for the LDA/SVM baseline.
    """
    mavFeatures = np.mean(np.abs(signalWindow), axis=0)
    return mavFeatures


def LoadAndProcess(dataPath, windowSize, overlapSize):
    """
    Loads one or more Ninapro files, applies a sliding window, and extracts MAV features.
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
        return np.array([]), np.array([])

    if not fileList:
        print(f"Warning: No .mat files found in {dataPath}. Check your directory structure.")
        return np.array([]), np.array([])
        
    allFeatures = []
    allLabels = []
    
    for filePath in fileList:
        print(f"Processing file: {os.path.basename(filePath)}...")
        mat = scipy.io.loadmat(filePath)
        
        emgSignals = mat['emg'] # (samples, channels)
        labels = mat['stimulus'].flatten() 
        
        stepSize = windowSize - overlapSize
        numSamples = emgSignals.shape[0]

        # Apply Sliding Window and Feature Extraction
        for i in range(0, numSamples - windowSize + 1, stepSize):
            windowStart = i
            windowEnd = i + windowSize
            
            windowedSignal = emgSignals[windowStart:windowEnd, :]
            
            windowLabels = labels[windowStart:windowEnd]
            (uniqueLabels, counts) = np.unique(windowLabels, return_counts=True)
            mostFrequentLabel = uniqueLabels[np.argmax(counts)]

            mavFeature = ExtractMavFeatures(windowedSignal) 
            allFeatures.append(mavFeature)
            allLabels.append(mostFrequentLabel)

    X = np.array(allFeatures)
    Y = np.array(allLabels)

    print(f"\nSuccessfully processed {len(fileList)} file(s).")
    print(f"Final Dataset Shape: Features (X)={X.shape}, Labels (y)={Y.shape}")
    return X, Y

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
        XFeaturesAll, yLabelsAll = LoadAndProcess(str(subjectDir), WINDOW_SIZE, OVERLAP_SIZE)

        print("\nExample 2: Loading ONLY the Grasping Primitives (E2)")
        singleFile = subjectDir / 'S1_A1_E2.mat'
        XFeaturesE2, yLabelsE2 = LoadAndProcess(str(singleFile), WINDOW_SIZE, OVERLAP_SIZE)