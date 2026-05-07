import os
import cv2
import csv
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# =========================
# CONFIG
# =========================
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
SEED = 42

MVTEC_CATEGORIES = ["hazelnut", "bottle"]
USE_CIFAR = True

COMBINED_PIXEL_W = 0.4
COMBINED_FEATURE_W = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# HELPER
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_valid_file(path):
    return os.path.isfile(path) and not os.path.basename(path).startswith(".")


def is_valid_dir(path):
    return os.path.isdir(path) and not os.path.basename(path).startswith(".")


def preprocess_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def normalize_by_ref(scores, ref_scores):
    mn = float(np.min(ref_scores))
    mx = float(np.max(ref_scores))
    return (scores - mn) / (mx - mn + 1e-8), mn, mx


def normalize_with_minmax(score, mn, mx):
    return (score - mn) / (mx - mn + 1e-8)


# =========================
# MODEL 1 - BASELINE
# =========================
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_features(self, x):
        return self.encoder(x)


# =========================
# MODEL 2 - ADVANCED
# =========================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(ch // r, 4)

        self.fc = nn.Sequential(
            nn.Linear(ch, mid),
            nn.ReLU(),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
        )

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.net(x))


class BottleneckAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()

        mid = max(ch // 8, 8)

        self.q = nn.Conv2d(ch, mid, 1)
        self.k = nn.Conv2d(ch, mid, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.k(x).view(b, -1, h * w)

        attn = F.softmax(torch.bmm(q, k) / (q.shape[-1] ** 0.5), dim=-1)

        v = self.v(x).view(b, c, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)

        return x + self.gamma * self.out(out)


class AttentionAE(nn.Module):
    """
    Denoising Attention Residual Autoencoder.
    Không dùng skip connection để tránh copy ảnh gốc.
    """
    def __init__(self, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            ResBlock(32),
            SEBlock(32),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResBlock(64),
            SEBlock(64),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResBlock(128),
            SEBlock(128),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResBlock(256),
            SEBlock(256),
        )

        self.bottleneck = BottleneckAttention(256)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def _add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def forward(self, x, add_noise=False):
        if add_noise and self.training:
            x = self._add_noise(x)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z = self.bottleneck(e4)

        d = self.dec4(z)
        d = self.dec3(d)
        d = self.dec2(d)

        return self.dec1(d)

    def get_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return self.bottleneck(e4)

    def get_multiscale_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.bottleneck(e4)
        return e1, e2, e3, z


# =========================
# LOSSES
# =========================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
        super().__init__()

        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        k2d = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", k2d.expand(3, 1, window_size, window_size))

    def forward(self, x, y):
        pad = self.window_size // 2

        mu_x = F.conv2d(x, self.kernel, padding=pad, groups=3)
        mu_y = F.conv2d(y, self.kernel, padding=pad, groups=3)

        sig_x = F.conv2d(x * x, self.kernel, padding=pad, groups=3) - mu_x ** 2
        sig_y = F.conv2d(y * y, self.kernel, padding=pad, groups=3) - mu_y ** 2
        sig_xy = F.conv2d(x * y, self.kernel, padding=pad, groups=3) - mu_x * mu_y

        num = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sig_x + sig_y + self.C2)

        ssim = num / (den + 1e-8)
        return 1.0 - ssim.mean()


class PerceptualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(weights=None)

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.65, beta=0.25, gamma=0.10):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualFeatureExtractor().to(DEVICE)
        self.perceptual.eval()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)

        with torch.no_grad():
            target_feat = self.perceptual(target)

        pred_feat = self.perceptual(pred)
        perceptual_loss = F.mse_loss(pred_feat, target_feat)

        return (
            self.alpha * mse_loss +
            self.beta * ssim_loss +
            self.gamma * perceptual_loss
        )


# =========================
# DATASET
# =========================
class NormalDataset(Dataset):
    def __init__(self, images, augment=False):
        self.images = images

        if augment:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1
                ),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.tf(self.images[idx])


# =========================
# DATA ANALYSIS
# =========================
def analyze_mvtec(category):
    train_dir = f"dataset/{category}/train/good"
    test_dir = f"dataset/{category}/test"

    info = {
        "dataset": category,
        "train_good": 0,
        "val_good": 0,
        "test_total": 0,
        "test_breakdown": {},
        "notes": [
            "Train chỉ có ảnh bình thường nên đây là bài toán one-class anomaly detection.",
            "Test có cả ảnh good và nhiều dạng defect nên dữ liệu bị mất cân bằng.",
            "Một số lỗi có kích thước nhỏ, khó phát hiện bằng pixel MSE đơn thuần.",
            "Ảnh MVTec có chi tiết bề mặt phức tạp, cần so sánh pixel score, feature score và combined score.",
        ]
    }

    if is_valid_dir(train_dir):
        info["train_good"] = len([
            f for f in os.listdir(train_dir)
            if is_valid_file(os.path.join(train_dir, f))
        ])

    if is_valid_dir(test_dir):
        for defect in sorted(os.listdir(test_dir)):
            d = os.path.join(test_dir, defect)

            if not is_valid_dir(d):
                continue

            cnt = len([
                f for f in os.listdir(d)
                if is_valid_file(os.path.join(d, f))
            ])

            info["test_breakdown"][defect] = cnt
            info["test_total"] += cnt

    return info


def analyze_cifar(normal_class=0):
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    info = {
        "dataset": "cifar",
        "normal_class": class_names[normal_class],
        "train_good": 0,
        "val_good": 0,
        "test_total": 0,
        "test_breakdown": {},
        "notes": [
            "CIFAR-10 one-class là dataset tùy chọn.",
            "Ảnh gốc 32x32 khi resize 256x256 có thể bị mờ.",
            "Một số class anomaly có hình dạng gần giống normal class.",
        ]
    }

    for i in range(1, 6):
        p = f"dataset/cifar-10-batches-py/data_batch_{i}"

        if not is_valid_file(p):
            continue

        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
            info["train_good"] += sum(1 for l in d[b"labels"] if l == normal_class)

    p = "dataset/cifar-10-batches-py/test_batch"

    if is_valid_file(p):
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")

            for l in d[b"labels"]:
                k = "good" if l == normal_class else class_names[l]
                info["test_breakdown"][k] = info["test_breakdown"].get(k, 0) + 1
                info["test_total"] += 1

    return info


# =========================
# LOAD DATA
# =========================
def load_mvtec(category, val_split=0.2):
    train_dir = f"dataset/{category}/train/good"
    test_dir = f"dataset/{category}/test"

    train_imgs = []
    test_imgs = []
    test_labels = []

    if is_valid_dir(train_dir):
        for f in sorted(os.listdir(train_dir)):
            p = os.path.join(train_dir, f)

            if not is_valid_file(p):
                continue

            img = cv2.imread(p)

            if img is not None:
                train_imgs.append(preprocess_bgr(img))

    if is_valid_dir(test_dir):
        for defect in sorted(os.listdir(test_dir)):
            dd = os.path.join(test_dir, defect)

            if not is_valid_dir(dd):
                continue

            for f in sorted(os.listdir(dd)):
                p = os.path.join(dd, f)

                if not is_valid_file(p):
                    continue

                img = cv2.imread(p)

                if img is not None:
                    test_imgs.append(preprocess_bgr(img))
                    test_labels.append(0 if defect == "good" else 1)

    if len(train_imgs) == 0:
        raise RuntimeError(f"Không tìm thấy train image trong: {train_dir}")

    if len(test_imgs) == 0:
        raise RuntimeError(f"Không tìm thấy test image trong: {test_dir}")

    train_imgs, val_imgs = train_test_split(
        train_imgs,
        test_size=val_split,
        random_state=SEED
    )

    return (
        np.array(train_imgs),
        np.array(val_imgs),
        np.array(test_imgs),
        np.array(test_labels)
    )


def load_cifar(normal_class=0, val_split=0.2):
    train_imgs = []
    test_imgs = []
    test_labels = []

    for i in range(1, 6):
        p = f"dataset/cifar-10-batches-py/data_batch_{i}"

        if not is_valid_file(p):
            continue

        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")

            for img, l in zip(d[b"data"], d[b"labels"]):
                if l == normal_class:
                    img = img.reshape(3, 32, 32).transpose(1, 2, 0)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    train_imgs.append(img)

    p = "dataset/cifar-10-batches-py/test_batch"

    if is_valid_file(p):
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")

            for img, l in zip(d[b"data"], d[b"labels"]):
                img = img.reshape(3, 32, 32).transpose(1, 2, 0)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                test_imgs.append(img)
                test_labels.append(0 if l == normal_class else 1)

    if len(train_imgs) == 0:
        raise RuntimeError("Không tìm thấy CIFAR train data trong dataset/cifar-10-batches-py")

    if len(test_imgs) == 0:
        raise RuntimeError("Không tìm thấy CIFAR test data trong dataset/cifar-10-batches-py")

    train_imgs, val_imgs = train_test_split(
        train_imgs,
        test_size=val_split,
        random_state=SEED
    )

    return (
        np.array(train_imgs),
        np.array(val_imgs),
        np.array(test_imgs),
        np.array(test_labels)
    )


# =========================
# MODEL UTILS
# =========================
def build_model(model_name):
    if model_name == "baseline":
        return ConvAutoEncoder().to(DEVICE)

    return AttentionAE().to(DEVICE)


def forward_model(model, model_name, x, train_mode=False):
    if model_name == "advanced":
        return model(x, add_noise=train_mode)

    return model(x)


def build_criterion(model_name):
    if model_name == "advanced":
        return CombinedLoss(alpha=0.65, beta=0.25, gamma=0.10).to(DEVICE)

    return nn.MSELoss()


def build_optimizer(model, optimizer_name, weight_decay=0.0):
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=weight_decay
        )

    return optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=weight_decay
    )


# =========================
# SCORE
# =========================
def compute_scores(model, model_name, images, return_errmaps=False):
    tf = transforms.ToTensor()
    model.eval()

    is_adv = model_name == "advanced"
    ms_w = [0.2, 0.3, 0.5]

    perceptual_extractor = None

    if is_adv:
        perceptual_extractor = PerceptualFeatureExtractor().to(DEVICE)
        perceptual_extractor.eval()

    pixel_scores = []
    feature_scores = []
    err_maps = []

    with torch.no_grad():
        for img in images:
            x = tf(img).unsqueeze(0).to(DEVICE)
            out = forward_model(model, model_name, x, train_mode=False)

            diff = (out - x) ** 2
            err_map = diff.squeeze().cpu().numpy().mean(axis=0)
            pixel_score = float(diff.mean().item())

            if is_adv:
                fx = model.get_multiscale_features(x)
                fo = model.get_multiscale_features(out)

                multi_scale_score = sum(
                    ms_w[i] * float(((fx[i + 1] - fo[i + 1]) ** 2).mean().item())
                    for i in range(3)
                )

                px = perceptual_extractor(x)
                po = perceptual_extractor(out)
                perceptual_score = float(((px - po) ** 2).mean().item())

                feature_score = 0.6 * multi_scale_score + 0.4 * perceptual_score

            else:
                fxb = model.get_features(x)
                fob = model.get_features(out)
                feature_score = float(((fxb - fob) ** 2).mean().item())

            pixel_scores.append(pixel_score)
            feature_scores.append(feature_score)

            if return_errmaps:
                err_maps.append(err_map)

    if return_errmaps:
        return (
            np.array(pixel_scores),
            np.array(feature_scores),
            np.array(err_maps)
        )

    return np.array(pixel_scores), np.array(feature_scores)


# =========================
# THRESHOLD + METRICS
# =========================
def find_best_threshold_youden(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def find_val_threshold(val_scores, percentile=95):
    return float(np.percentile(val_scores, percentile))


def find_best_threshold_f1(scores, labels):
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 300)

    best_thr = thresholds[0]
    best_f1 = -1

    for thr in thresholds:
        preds = (scores > thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_thr), float(best_f1)


def calc_classification_metrics(scores, labels, threshold):
    preds = (scores > threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def calc_auc_metrics(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)

    auroc = float(auc(fpr, tpr))
    prauc = float(auc(rec[::-1], prec[::-1]))

    return auroc, prauc, fpr, tpr, prec, rec


# =========================
# PLOTS - CHỈ GIỮ BIỂU ĐỒ CẦN THIẾT
# =========================
def save_learning_curve(train_losses, val_losses, path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_roc_pr(fpr, tpr, prec, rec, auroc, prauc, path_prefix):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC={auroc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_roc.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f"PR-AUC={prauc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_pr.png", dpi=120)
    plt.close()


def save_sample_heatmaps(model, model_name, images, labels, out_dir, max_samples=8):
    ensure_dir(out_dir)

    tf = transforms.ToTensor()
    saved = 0

    model.eval()

    with torch.no_grad():
        for img, label in zip(images, labels):
            if saved >= max_samples:
                break

            x = tf(img).unsqueeze(0).to(DEVICE)
            out = forward_model(model, model_name, x, train_mode=False)

            diff = (out - x) ** 2
            err_map = diff.squeeze().cpu().numpy().mean(axis=0)

            heat = (err_map - err_map.min()) / (err_map.max() - err_map.min() + 1e-8)
            heat = cv2.GaussianBlur((heat * 255).astype(np.uint8), (11, 11), 0)

            heat_c = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr, 0.55, heat_c, 0.45, 0)

            recon = out.squeeze().cpu().permute(1, 2, 0).numpy()
            recon = cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            tag = "anomaly" if label == 1 else "normal"

            cv2.imwrite(os.path.join(out_dir, f"{saved}_{tag}_input.png"), img_bgr)
            cv2.imwrite(os.path.join(out_dir, f"{saved}_{tag}_recon.png"), recon)
            cv2.imwrite(os.path.join(out_dir, f"{saved}_{tag}_heatmap.png"), heat_c)
            cv2.imwrite(os.path.join(out_dir, f"{saved}_{tag}_overlay.png"), overlay)

            saved += 1


# =========================
# TRAINING
# =========================
def train_experiment(
    dataset_name,
    train_imgs,
    val_imgs,
    test_imgs,
    test_labels,
    model_name,
    optimizer_name,
    regularization_name,
    weight_decay=0.0,
    use_augmentation=False,
    progress_callback=None,
):
    exp_name = f"{dataset_name}_{model_name}_{optimizer_name}_{regularization_name}"
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    ensure_dir(exp_dir)

    log_path = os.path.join(exp_dir, "train_log.json")

    print(f"\n{'=' * 70}")
    print(f"TRAIN: {exp_name}")
    print(f"{'=' * 70}")

    model = build_model(model_name)
    optimizer = build_optimizer(model, optimizer_name, weight_decay)
    criterion = build_criterion(model_name)

    params = count_parameters(model)

    training_history = {
        "exp_name": exp_name,
        "dataset": dataset_name,
        "model_name": model_name,
        "optimizer": optimizer_name,
        "regularization": regularization_name,
        "weight_decay": weight_decay,
        "augmentation": use_augmentation,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "params": params,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "epoch_logs": [],
        "train_losses": [],
        "val_losses": [],
    }

    print(f"Params: {params:,}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6
    )

    train_loader = DataLoader(
        NormalDataset(train_imgs, augment=use_augmentation),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    mse_criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = os.path.join(exp_dir, "model_best.pth")

    for epoch in range(EPOCHS):
        model.train()
        total_train = 0.0

        for x in train_loader:
            x = x.to(DEVICE)

            out = forward_model(model, model_name, x, train_mode=True)
            loss = criterion(out, x)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0

        with torch.no_grad():
            for img in val_imgs:
                x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
                out = forward_model(model, model_name, x, train_mode=False)
                total_val += mse_criterion(out, x).item()

        avg_val = total_val / len(val_imgs)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)

        log_line = (
            f"[{exp_name}] epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train={avg_train:.6f} | "
            f"val={avg_val:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        print(log_line)

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": optimizer.param_groups[0]["lr"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        training_history["epoch_logs"].append(epoch_entry)
        training_history["train_losses"].append(avg_train)
        training_history["val_losses"].append(avg_val)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)

        if progress_callback:
            progress_callback(
                epoch + 1,
                EPOCHS,
                avg_train,
                avg_val,
                log_line
            )

    training_history["finished_at"] = datetime.now().isoformat(timespec="seconds")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best model with val loss = {best_val_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pth"))

    # =========================
    # EVALUATION
    # =========================
    val_pixel, val_feature = compute_scores(
        model,
        model_name,
        val_imgs,
        return_errmaps=False
    )

    test_pixel, test_feature, test_errmaps = compute_scores(
        model,
        model_name,
        test_imgs,
        return_errmaps=True
    )

    val_pixel_n, val_pixel_min, val_pixel_max = normalize_by_ref(val_pixel, val_pixel)
    test_pixel_n, _, _ = normalize_by_ref(test_pixel, val_pixel)

    val_feature_n, val_feature_min, val_feature_max = normalize_by_ref(val_feature, val_feature)
    test_feature_n, _, _ = normalize_by_ref(test_feature, val_feature)

    val_combined = COMBINED_PIXEL_W * val_pixel_n + COMBINED_FEATURE_W * val_feature_n
    test_combined = COMBINED_PIXEL_W * test_pixel_n + COMBINED_FEATURE_W * test_feature_n

    threshold_pixel_val = find_val_threshold(val_pixel, percentile=95)
    threshold_feature_val = find_val_threshold(val_feature, percentile=95)
    threshold_combined_val = find_val_threshold(val_combined, percentile=95)

    threshold_pixel_youden = find_best_threshold_youden(test_pixel, test_labels)
    threshold_feature_youden = find_best_threshold_youden(test_feature, test_labels)
    threshold_combined_youden = find_best_threshold_youden(test_combined, test_labels)

    threshold_pixel_f1, best_f1_pixel = find_best_threshold_f1(test_pixel, test_labels)
    threshold_feature_f1, best_f1_feature = find_best_threshold_f1(test_feature, test_labels)
    threshold_combined_f1, best_f1_combined = find_best_threshold_f1(test_combined, test_labels)

    auroc_pixel, prauc_pixel, fpr_p, tpr_p, prec_p, rec_p = calc_auc_metrics(test_pixel, test_labels)
    auroc_feature, prauc_feature, fpr_f, tpr_f, prec_f, rec_f = calc_auc_metrics(test_feature, test_labels)
    auroc_combined, prauc_combined, fpr_c, tpr_c, prec_c, rec_c = calc_auc_metrics(test_combined, test_labels)

    class_pixel = calc_classification_metrics(test_pixel, test_labels, threshold_pixel_f1)
    class_feature = calc_classification_metrics(test_feature, test_labels, threshold_feature_f1)
    class_combined = calc_classification_metrics(test_combined, test_labels, threshold_combined_f1)

    print(f"AUROC pixel={auroc_pixel:.4f} | feature={auroc_feature:.4f} | combined={auroc_combined:.4f}")
    print(f"PR-AUC pixel={prauc_pixel:.4f} | feature={prauc_feature:.4f} | combined={prauc_combined:.4f}")
    print(f"Best F1 combined={best_f1_combined:.4f}")

    # =========================
    # SAVE PLOTS
    # =========================
    save_learning_curve(
        train_losses,
        val_losses,
        os.path.join(exp_dir, "learning_curve.png")
    )

    save_roc_pr(
        fpr_p,
        tpr_p,
        prec_p,
        rec_p,
        auroc_pixel,
        prauc_pixel,
        os.path.join(exp_dir, "pixel")
    )

    save_roc_pr(
        fpr_f,
        tpr_f,
        prec_f,
        rec_f,
        auroc_feature,
        prauc_feature,
        os.path.join(exp_dir, "feature")
    )

    save_roc_pr(
        fpr_c,
        tpr_c,
        prec_c,
        rec_c,
        auroc_combined,
        prauc_combined,
        os.path.join(exp_dir, "combined")
    )

    save_sample_heatmaps(
        model,
        model_name,
        test_imgs,
        test_labels,
        os.path.join(exp_dir, "sample_heatmaps"),
        max_samples=8
    )

    metrics = {
        "dataset": dataset_name,
        "model_name": model_name,
        "optimizer": optimizer_name,
        "regularization": regularization_name,
        "weight_decay": weight_decay,
        "augmentation": use_augmentation,
        "params": count_parameters(model),

        "threshold_pixel_val95": threshold_pixel_val,
        "threshold_feature_val95": threshold_feature_val,
        "threshold_combined_val95": threshold_combined_val,

        "threshold_pixel_youden": threshold_pixel_youden,
        "threshold_feature_youden": threshold_feature_youden,
        "threshold_combined_youden": threshold_combined_youden,

        "threshold_pixel_f1": threshold_pixel_f1,
        "threshold_feature_f1": threshold_feature_f1,
        "threshold_combined_f1": threshold_combined_f1,

        "threshold_pixel": threshold_pixel_f1,
        "threshold_feature": threshold_feature_f1,
        "threshold_combined": threshold_combined_f1,

        "selected_threshold_method": "combined_f1",
        "selected_score": "combined_score",
        "combined_pixel_weight": COMBINED_PIXEL_W,
        "combined_feature_weight": COMBINED_FEATURE_W,

        "pixel_direction": "higher_is_anomaly",
        "feature_direction": "higher_is_anomaly",
        "combined_direction": "higher_is_anomaly",

        "auroc_pixel": auroc_pixel,
        "prauc_pixel": prauc_pixel,
        "auroc_feature": auroc_feature,
        "prauc_feature": prauc_feature,
        "auroc_combined": auroc_combined,
        "prauc_combined": prauc_combined,

        "best_f1_pixel": best_f1_pixel,
        "best_f1_feature": best_f1_feature,
        "best_f1_combined": best_f1_combined,

        "accuracy_pixel_f1": class_pixel["accuracy"],
        "precision_pixel_f1": class_pixel["precision"],
        "recall_pixel_f1": class_pixel["recall"],
        "f1_pixel_f1": class_pixel["f1"],

        "accuracy_feature_f1": class_feature["accuracy"],
        "precision_feature_f1": class_feature["precision"],
        "recall_feature_f1": class_feature["recall"],
        "f1_feature_f1": class_feature["f1"],

        "accuracy_combined_f1": class_combined["accuracy"],
        "precision_combined_f1": class_combined["precision"],
        "recall_combined_f1": class_combined["recall"],
        "f1_combined_f1": class_combined["f1"],

        "train_loss_last": float(train_losses[-1]),
        "val_loss_last": float(val_losses[-1]),
        "best_val_loss": float(best_val_loss),

        "val_pixel_min": float(val_pixel_min),
        "val_pixel_max": float(val_pixel_max),
        "val_feature_min": float(val_feature_min),
        "val_feature_max": float(val_feature_max),

        "advanced_description": (
            "Denoising Attention Residual Autoencoder with "
            "SE Block, Bottleneck Self-Attention, MSE+SSIM+Perceptual Loss"
        ) if model_name == "advanced" else "Convolutional Autoencoder baseline"
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    training_history["metrics"] = metrics

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)

    return metrics


# =========================
# SUMMARY
# =========================
def save_summary_csv(rows, path):
    if not rows:
        return

    keys = [
        "dataset",
        "model_name",
        "optimizer",
        "regularization",
        "weight_decay",
        "augmentation",
        "params",

        "threshold_pixel",
        "threshold_feature",
        "threshold_combined",
        "selected_threshold_method",
        "selected_score",

        "auroc_pixel",
        "prauc_pixel",
        "auroc_feature",
        "prauc_feature",
        "auroc_combined",
        "prauc_combined",

        "best_f1_pixel",
        "best_f1_feature",
        "best_f1_combined",

        "accuracy_combined_f1",
        "precision_combined_f1",
        "recall_combined_f1",
        "f1_combined_f1",

        "train_loss_last",
        "val_loss_last",
        "best_val_loss",

        "advanced_description",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    summary_rows = []
    dataset_reports = []

    experiment_configs = [
        # Hai mô hình chính
        {
            "model_name": "baseline",
            "optimizer": "adam",
            "regularization": "weight_decay",
            "weight_decay": 1e-4,
            "aug": False,
        },
        {
            "model_name": "advanced",
            "optimizer": "adam",
            "regularization": "weight_decay",
            "weight_decay": 1e-4,
            "aug": False,
        },

        # So sánh optimizer
        {
            "model_name": "baseline",
            "optimizer": "sgd",
            "regularization": "weight_decay",
            "weight_decay": 1e-4,
            "aug": False,
        },

        # So sánh regularization
        {
            "model_name": "baseline",
            "optimizer": "adam",
            "regularization": "augmentation",
            "weight_decay": 0.0,
            "aug": True,
        },
    ]

    for category in MVTEC_CATEGORIES:
        report = analyze_mvtec(category)

        train_imgs, val_imgs, test_imgs, test_labels = load_mvtec(category)

        report["val_good"] = len(val_imgs)
        dataset_reports.append(report)

        n_normal = int((test_labels == 0).sum())
        n_anomaly = int((test_labels == 1).sum())

        print(
            f"\n[{category}] "
            f"train={len(train_imgs)} | "
            f"val={len(val_imgs)} | "
            f"test={len(test_imgs)} "
            f"(normal={n_normal}, anomaly={n_anomaly})"
        )

        for cfg in experiment_configs:
            row = train_experiment(
                dataset_name=category,
                train_imgs=train_imgs,
                val_imgs=val_imgs,
                test_imgs=test_imgs,
                test_labels=test_labels,

                model_name=cfg["model_name"],
                optimizer_name=cfg["optimizer"],
                regularization_name=cfg["regularization"],
                weight_decay=cfg["weight_decay"],
                use_augmentation=cfg["aug"],
            )

            summary_rows.append(row)

    if USE_CIFAR:
        report = analyze_cifar(normal_class=0)

        train_imgs, val_imgs, test_imgs, test_labels = load_cifar(normal_class=0)

        report["val_good"] = len(val_imgs)
        dataset_reports.append(report)

        n_normal = int((test_labels == 0).sum())
        n_anomaly = int((test_labels == 1).sum())

        print(
            f"\n[cifar] "
            f"train={len(train_imgs)} | "
            f"val={len(val_imgs)} | "
            f"test={len(test_imgs)} "
            f"(normal={n_normal}, anomaly={n_anomaly})"
        )

        for cfg in experiment_configs:
            row = train_experiment(
                dataset_name="cifar",
                train_imgs=train_imgs,
                val_imgs=val_imgs,
                test_imgs=test_imgs,
                test_labels=test_labels,

                model_name=cfg["model_name"],
                optimizer_name=cfg["optimizer"],
                regularization_name=cfg["regularization"],
                weight_decay=cfg["weight_decay"],
                use_augmentation=cfg["aug"],
            )

            summary_rows.append(row)

    save_summary_csv(
        summary_rows,
        os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    )

    with open(os.path.join(OUTPUT_DIR, "dataset_reports.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_reports, f, indent=2, ensure_ascii=False)

    print("\nĐã train xong tất cả thí nghiệm.")
    print(f"Kết quả lưu tại: {OUTPUT_DIR}/")