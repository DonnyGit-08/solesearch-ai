"""Loads the trained EfficientNet-B0 and runs predictions on sneaker photos."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


class SneakerModel:
    def __init__(self, model_path: Path):
        # weights_only=True only loads tensors and plain values, never arbitrary code
        package = torch.load(model_path, map_location="cpu", weights_only=True)
        self.class_names = list(package["class_names"])
        self.threshold = float(package.get("confidence_threshold", 0.6))

        net = efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(self.class_names))
        net.load_state_dict(package["state_dict"])
        net.eval()
        self.net = net

        size = int(package.get("img_size", 224))
        self.tf = transforms.Compose([
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(package["mean"], package["std"]),
        ])

    @torch.no_grad()
    def analyze(self, img: Image.Image):
        """Returns (class probabilities, normalized 1280-number embedding) for one image."""
        x = self.tf(img.convert("RGB")).unsqueeze(0)
        feats = self.net.avgpool(self.net.features(x)).flatten(1)  # the "fingerprint" of the image
        probs = torch.softmax(self.net.classifier(feats), dim=1)[0].numpy()
        emb = feats[0].numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return probs, emb

    def top_k(self, probs, k=3):
        order = np.argsort(probs)[::-1][:k]
        return [(self.class_names[i], float(probs[i])) for i in order]
