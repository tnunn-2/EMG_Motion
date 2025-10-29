import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loader import LoadAndProcess
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def BuildRestData(gestureDict):
    """
    Creates xRest and yRest given a gesture from data_loader.LoadAndProcess()
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
    restClf = RandomForestClassifier(n_estimators=100)
    restClf.fit(xTrainRest, yTrainRest)

    xActive, yActive = BuildActiveData(gestureDict)
    xTrainActive, xTestActive, yTrainActive, yTestActive = train_test_split(xActive, yActive, test_size = 0.2)
    #scaler = StandardScaler()
    #gestureClf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    gestureClf, scaler = tuneSVC(xTrainActive, yTrainActive)
    #xTrainScaled = scaler.fit_transform(xTrainActive)
    xTestScaled = scaler.transform(xTestActive)
    #gestureClf.fit(xTrainScaled, yTrainActive)

    yPredGesture = gestureClf.predict(xTestScaled)
    yPredRest = restClf.predict(xTestRest)

    print("Rest Detector Accuracy:{:.4f}".format(accuracy_score(yTestRest, yPredRest)))
    print("Gesture Classifier (SVC) Accuracy:{:.4f}".format(accuracy_score(yTestActive, yPredGesture)))

    print("\nSample Gesture Predictions (True → Predicted):")
    for i in range(10):
        print(f"  {int(yTestActive[i]):>2} → {int(yPredGesture[i])}")


    return restClf, gestureClf, scaler
    

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

    grid = GridSearchCV(pipe, paramGrid, cv=3, n_jobs=1, verbose=1)
    grid.fit(xActive, yActive)

    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

    # Extract trained model and scaler
    bestModel = grid.best_estimator_
    scaler = bestModel.named_steps['scaler']
    gestureClf = bestModel.named_steps['svc']

    return gestureClf, scaler


#Test
if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).parent.parent 
    BASE_DATA = PROJECT_ROOT / 'data' / 'ninapro' / 'DB1'
    subjectDir = BASE_DATA / 'S1'
    singleFile = subjectDir / 'S1_A1_E2.mat'

    print("\nLoading EMG data from: ", singleFile)
    singleDict, xE2, yE2 = LoadAndProcess(str(singleFile))
    print("Gestures found:", list(singleDict.keys()))


    print("\nTraining model on data...")
    restClf, gestureClf, scaler = train(singleDict)

    testFeature = xE2[np.random.randint(0, len(xE2))]
    restPred = restClf.predict([testFeature])[0]

    if restPred == 0:
        print("\nDetected Rest")
    else:
        gesturePred = gestureClf.predict(scaler.transform([testFeature]))[0]
        print("\nDetected Gesture ID: ", gesturePred)
    