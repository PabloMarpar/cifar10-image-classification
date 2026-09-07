"""Entrena el modelo baseline (MLP totalmente conectada) sobre CIFAR-10 sin data augmentation."""

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate import collect_predictions, plot_confusion_matrix
from models import SimpleMLP

OUTPUT_DIR = "outputs"
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Usando dispositivo: {DEVICE}")

    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = SimpleMLP().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, loss_fn, optimizer)
        test_loss, accuracy = evaluate_loss_accuracy(model, test_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {test_loss:.4f} - accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/baseline_mlp.pth")

    y_true, y_pred = collect_predictions(model, test_loader, DEVICE)
    plot_confusion_matrix(
        y_true, y_pred, "Matriz de confusión - MLP baseline", f"{OUTPUT_DIR}/confusion_matrix_mlp.png"
    )


if __name__ == "__main__":
    main()
