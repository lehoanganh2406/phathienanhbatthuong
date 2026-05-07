from PIL import Image

img = Image.open("dataset/hazelnut/train/good/000.png")
print(img.size)   # (width, height)
import torch
import torch.nn as nn


# =========================
# MODEL 1
# =========================
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_features(self, x):
        return self.encoder(x)


# =========================
# MODEL 2
# =========================
class DenoisingAutoEncoder(nn.Module):
    def __init__(self, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, add_noise=False):
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0.0, 1.0)
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_features(self, x):
        return self.encoder(x)


# =========================
# HÀM IN SHAPE
# =========================
def print_model_shapes(model, model_name, input_shape=(1, 3, 256, 256)):
    x = torch.randn(input_shape)

    print(f"\n{'='*20} {model_name} {'='*20}")
    print("Input :", f"{x.shape[1]}x{x.shape[2]}x{x.shape[3]}")

    for i, layer in enumerate(model.encoder):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            print(f"Encoder {i}: {x.shape[1]}x{x.shape[2]}x{x.shape[3]}")

    for i, layer in enumerate(model.decoder):
        x = layer(x)
        if isinstance(layer, nn.ConvTranspose2d):
            print(f"Decoder {i}: {x.shape[1]}x{x.shape[2]}x{x.shape[3]}")

    print("Output :", f"{x.shape[1]}x{x.shape[2]}x{x.shape[3]}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model1 = ConvAutoEncoder()
    model2 = DenoisingAutoEncoder()

    print_model_shapes(model1, "ConvAutoEncoder")
    print_model_shapes(model2, "DenoisingAutoEncoder")