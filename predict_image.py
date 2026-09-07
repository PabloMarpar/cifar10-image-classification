"""Clasifica una o varias imágenes propias con un modelo ya entrenado.

Ejemplo:
    python predict_image.py --arch cnn --model outputs/cnn_augmented.pth foto1.jpg foto2.jpg
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from evaluate import CLASSES
from models import AugmentedCNN, SimpleMLP

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

ARCHS = {"mlp": SimpleMLP, "cnn": AugmentedCNN}


def load_model(arch: str, checkpoint_path: str, device: str):
    model = ARCHS[arch]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict(model, image_path: str, device: str) -> str:
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return CLASSES[output.argmax(1).item()]


def main():
    parser = argparse.ArgumentParser(description="Clasifica imágenes con un modelo CIFAR-10 ya entrenado.")
    parser.add_argument("images", nargs="+", help="Rutas a las imágenes a clasificar")
    parser.add_argument("--arch", choices=ARCHS.keys(), default="cnn")
    parser.add_argument("--model", default="outputs/cnn_augmented.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.arch, args.model, device)

    for image_path in args.images:
        print(f"{image_path}: {predict(model, image_path, device)}")


if __name__ == "__main__":
    main()
