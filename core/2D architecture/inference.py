import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .model import TumorNet34Bayes
from .utils import GradCAMpp

def preprocess_image_bgr(path, img_size=224):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1)) 
    return img

@torch.no_grad()
def mc_dropout_inference(model, x, T=20, device="cpu"):
    model.train()  
    preds = []
    logvars = []
    for _ in range(T):
        logits, logvar, _ = model(x)
        probs = torch.sigmoid(logits)
        preds.append(probs)
        logvars.append(logvar)
    preds = torch.stack(preds, dim=0)
    logvars = torch.stack(logvars, dim=0)  
    mean_prob = preds.mean(dim=0)  
    epistemic = preds.std(dim=0)   
    aleatoric = torch.exp(logvars).mean(dim=0).sqrt()
    total_unc = torch.sqrt(epistemic**2 + aleatoric**2 + 1e-8)
    return mean_prob.squeeze(1), epistemic.squeeze(1), aleatoric.squeeze(1), total_unc.squeeze(1)

def predict_single(
    image_path,
    checkpoint_path,
    device=None,
    img_size=224,
    mc_samples=20
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TumorNet34Bayes(dropout_p=0.4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    arr = preprocess_image_bgr(image_path, img_size)
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

    # MC Dropout pass
    mean_prob, epistemic, aleatoric, total_unc = mc_dropout_inference(model, x, T=mc_samples, device=device)

    # Grad-CAM++
    cam = None
    gc = GradCAMpp(model, target_layer="layer4")
    model.zero_grad(set_to_none=True)
    logits, _, _ = model(x)
    logits.sum().backward()
    cam = gc.generate(logits).squeeze(0).detach().cpu().numpy()  
    gc.remove()

    return {
        "prob": float(mean_prob.item()),
        "epistemic": float(epistemic.item()),
        "aleatoric": float(aleatoric.item()),
        "total_uncertainty": float(total_unc.item()),
        "cam": cam
    }
