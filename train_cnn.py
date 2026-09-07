"""Entrena una CNN con batch normalization, dropout y data augmentation sobre CIFAR-10,
combinando el dataset original con una copia aumentada (flips + random crop)."""

import json
import os

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from evaluate import collect_predictions, plot_confusion_matrix
from models import AugmentedCNN

OUTPUT_DIR = "outputs"
BATCH_SIZE = 64
# Reducido desde 20: sobre CPU, con el dataset duplicado por el augmentation,
# 20 épocas es poco práctico. Con GPU, sube este valor para mejor accuracy.
EPOCHS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate_loss_accuracy(model, dataloader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        total_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(dataloader), 100 * correct / total


def plot_training_curves(history):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="Train loss", color="#2a78d6")
    axes[0].plot(history["test_loss"], label="Test loss", color="#eb6834")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Evolución de la pérdida")
    axes[0].legend()

    axes[1].plot(history["test_accuracy"], color="#1baf7a")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy en test")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.close(fig)


def main(epochs: int = EPOCHS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Usando dispositivo: {DEVICE}")

    plain_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor())
    augmented_train = datasets.CIFAR10(root="data", train=True, download=True, transform=TRANSFORM_TRAIN)
    train_data = ConcatDataset([plain_train, augmented_train])
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=TRANSFORM_TEST)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = AugmentedCNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        test_loss, accuracy = evaluate_loss_accuracy(model, test_loader, loss_fn)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(accuracy)
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} "
              f"- test_loss: {test_loss:.4f} - accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/cnn_augmented.pth")

    with open(f"{OUTPUT_DIR}/history.json", "w") as fh:
        json.dump(history, fh)

    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/cnn_augmented.pth"))
    y_true, y_pred = collect_predictions(model, test_loader, DEVICE)
    plot_confusion_matrix(
        y_true, y_pred, "Matriz de confusión - CNN con augmentation", f"{OUTPUT_DIR}/confusion_matrix_cnn.png"
    )
    plot_training_curves(history)
    print(f"\nMejor accuracy en test: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
