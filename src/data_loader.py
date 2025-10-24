import numpy as np
import scipy.io
import os
from pathlib import Path

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
    
    dataPath: A string path. Can be:
        1. A path to a single .mat file (e.g., '.../S1_A1_E2.mat')
        2. A path to a directory containing .mat files (e.g., '.../S1')
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