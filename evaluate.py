"""Utilidades de evaluación compartidas: matriz de confusión y classification report."""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(classification_report(y_true, y_pred, target_names=CLASSES))
