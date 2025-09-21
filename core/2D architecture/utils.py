import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        loss = self.bce(logits, targets)
        pt = torch.exp(-loss)
        fl = self.alpha * (1 - pt) ** self.gamma * loss
        return fl.mean() if self.reduction == 'mean' else fl.sum()

class EarlyStoppingAUC:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -1.0
        self.count = 0
        self.should_stop = False

    def step(self, auc):
        if auc > self.best + self.min_delta:
            self.best = auc
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(np.int32)
    return {
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    }

def ece_score(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = (probs[mask] >= 0.5).astype(int).mean() if probs.ndim==1 else (probs[mask].argmax(1) == labels[mask]).mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.ElasticTransform(alpha=1.0, sigma=20, alpha_affine=10, p=0.5),
            A.GridDistortion(p=0.5)
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])

class GradCAMpp:
    def __init__(self, model, target_layer="layer4"):
        self.model = model
        self.model.eval()
        self.target_layer = dict([*self.model.named_modules()])[target_layer]
        self.activations = None
        self.gradients = None
        self._fh = self.target_layer.register_forward_hook(self._forward_hook)
        self._bh = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, logits):
        # logits is (B,1)
        assert self.activations is not None and self.gradients is not None
        grads = self.gradients  
        act = self.activations

        B, C, H, W = act.shape
        grads2 = grads ** 2
        grads3 = grads2 * grads

        # alpha_k (Grad-CAM++)
        denom = 2 * grads2 + torch.sum(act * grads3, dim=(2,3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads2 / denom  

        weights = torch.relu((alpha * torch.relu(grads)).sum(dim=(2,3)))  #
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * act).sum(dim=1)  
        cam = torch.relu(cam)
        cam = cam / (cam.amax(dim=(1,2), keepdim=True) + 1e-8)
        return cam.detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()
