from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import uuid


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", gesture_names=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if gesture_names is not None:
        plt.xticks(np.arange(len(gesture_names)) + 0.5, gesture_names, rotation=90)
        plt.yticks(np.arange(len(gesture_names)) + 0.5, gesture_names, rotation=0)

    plt.tight_layout()
    plt.show()

    return cm

def plot_loso_results(results):
    unique_id = uuid.uuid4()
    subjects = sorted(results["dann"].keys())

    dann_acc = [results["dann"][s] for s in subjects]
    proto_before = [results["finetune_dann"][s]["proto_before_acc"] for s in subjects]
    knn_before = [results["finetune_dann"][s]["knn_before_acc"] for s in subjects]

    proto_after = [
        results["finetune_dann"][s].get("proto_after_acc", None)
        for s in subjects
    ]
    knn_after = [
        results["finetune_dann"][s].get("knn_after_acc", None)
        for s in subjects
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(subjects, dann_acc, "-o", label="DANN", linewidth=2)
    plt.plot(subjects, proto_before, "-o", label="Proto (Before Finetune)")
    plt.plot(subjects, knn_before, "-o", label="kNN (Before Finetune)")

    if any(proto_after):
        plt.plot(subjects, proto_after, "-o", label="Proto (After Finetune)")
    if any(knn_after):
        plt.plot(subjects, knn_after, "-o", label="kNN (After Finetune)")

    plt.xlabel("Test Subject")
    plt.ylabel("Accuracy")
    plt.title("LOSO Performance Across Subjects")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f'figures/loso_results_{unique_id}.png')
    plt.show()
    plt.pause(5)
    plt.close()

def plot_single_subject_results(results, subj):
    unique_id = uuid.uuid4()

    metrics = [
        "DANN",
        "Proto Before Finetune",
        "kNN Before Finetune",
        "Proto After Finetune",
        "kNN After Finetune"
    ]
    values = [
        results["dann"][subj],
        results["finetune_dann"][subj]["proto_before_acc"],
        results["finetune_dann"][subj]["knn_before_acc"],
        results["finetune_dann"][subj].get("proto_after_acc", 0),
        results["finetune_dann"][subj].get("knn_after_acc", 0)
    ]

    plt.figure(figsize=(10,6))
    plt.title(f"Performance for Subject {subj}")
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.grid(True, axis='y')
    plt.savefig(f'figures/singleloso_results_subj{subj}_{unique_id}.png')
    plt.show()
    plt.pause(5)
    plt.close()


