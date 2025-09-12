#!/usr/bin/env python3
import sys, subprocess, pathlib, urllib.request, shutil, os

# ---------- install deps if missing ----------
def ensure(pkg_spec: str):
    try:
        __import__(pkg_spec.split("==")[0].split(">=")[0].split("[")[0])
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_spec], check=True)

ensure("opencv-python")
ensure("matplotlib")
ensure("scikit-learn")
# segment-anything is not on PyPI; install from GitHub if import fails
try:
    import segment_anything  # noqa
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "git+https://github.com/facebookresearch/segment-anything.git"], check=True)

# ---------- imports ----------
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.cluster import DBSCAN
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---------- resources ----------
ckpt_path = pathlib.Path("/home/beloslava/Downloads/sam_vit_h_4b8939.pth")  # your local file
if not ckpt_path.exists():
    # fallback: download to current dir if local path missing
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    print(f"[info] Local checkpoint not found at {ckpt_path}. Downloading to ./sam_vit_h_4b8939.pth ...")
    with urllib.request.urlopen(url) as r, open("sam_vit_h_4b8939.pth", "wb") as f:
        shutil.copyfileobj(r, f)
    ckpt_path = pathlib.Path("sam_vit_h_4b8939.pth")

img_path = pathlib.Path("/home/beloslava/Downloads/img2.jpeg")
if not img_path.exists():
    img_url = "https://i2.wp.com/lifemadesimplebakes.com/wp-content/uploads/2014/09/Classic-Pepperoni-Pizza-1.jpg"
    print(f"[info] Sample image not found at {img_path}. Downloading to ./sample_pizza.jpg ...")
    with urllib.request.urlopen(img_url) as r, open("sample_pizza.jpg", "wb") as f:
        shutil.copyfileobj(r, f)
    img_path = pathlib.Path("sample_pizza.jpg")

# ---------- SAM setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=51,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

def show_anns(anns):
    if len(anns) == 0:
        return
    anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    H, W = anns[0]["segmentation"].shape
    overlay = np.ones((H, W, 4), dtype=float); overlay[:, :, 3] = 0
    for ann in anns:
        m = ann["segmentation"]
        color = np.concatenate([np.random.random(3), [0.9]])
        overlay[m] = color
    ax.imshow(overlay)

# ---------- load + resize image ----------
img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
img = cv2.resize(img, (1024, 1024))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("IMG shape:", img_rgb.shape)

# ---------- run SAM ----------
# Access encoder features for your patch embedding trick
mask_generator.predictor.set_image(img_rgb)
enc_emb = mask_generator.predictor.features  # [1, C, H', W']
enc_emb = enc_emb.to("cpu").numpy()[0].transpose((1, 2, 0))  # -> (H', W', C)
print("Encoder embedding shape:", enc_emb.shape)

masks = mask_generator.generate(img_rgb)
masks = sorted(masks, key=lambda x: x["area"], reverse=True)
print("Number of masks:", len(masks))
print("Mask shape:", masks[0]["segmentation"].shape if masks else None)

plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
show_anns(masks)
plt.axis("off")
plt.tight_layout()
plt.show()

# ---------- mask → patch-embedding pooling ----------
# Assumes 1024x1024 input and ViT patch size 16 => 64x64 grid
def mask_to_mean_patch_embedding(mask_bool: np.ndarray, patch_feats: np.ndarray):
    # mask_bool: (1024, 1024) boolean array
    # patch_feats: (64, 64, C)
    H, W = mask_bool.shape
    assert (H, W) == (1024, 1024), "This pooling assumes 1024x1024 input."
    # Downsample mask to 64x64 by summing each 16x16 block
    m = mask_bool.astype(np.uint8)
    m = m.reshape(64, 16, 64, 16).sum(axis=(1, 3))  # -> (64, 64)
    sel = m > 1  # pick blocks with enough foreground
    if not np.any(sel):
        return patch_feats.mean(axis=(0, 1))
    return patch_feats[sel].mean(axis=0)

emb_list = []
for ann in masks:
    emb = mask_to_mean_patch_embedding(ann["segmentation"], enc_emb)
    emb_list.append(emb)
emb_arr = np.stack(emb_list, axis=0) if emb_list else np.empty((0, enc_emb.shape[-1]))
print("Mask embedding array:", emb_arr.shape)

# ---------- cluster with DBSCAN ----------
clustering = DBSCAN(eps=0.06, min_samples=8, metric="cosine").fit(emb_arr)
labels = clustering.labels_
print("Cluster labels:", np.unique(labels))

# Example: collect cluster 0 mask (if present)
if np.any(labels == 0):
    idxs = np.where(labels == 0)[0]
    print("Items in cluster 0:", idxs.shape[0])
    combined = np.zeros_like(masks[0]["segmentation"], dtype=np.uint8)
    for i in idxs:
        combined |= masks[i]["segmentation"].astype(np.uint8)
    # visualize
    masked = img_rgb.copy()
    masked[~combined.astype(bool)] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(combined, cmap="gray"); axes[1].set_title("Label 0 Semantic Map"); axes[1].axis("off")
    axes[2].imshow(masked); axes[2].set_title("Output"); axes[2].axis("off")
    plt.tight_layout(); plt.show()
else:
    print("No items in cluster 0; adjust DBSCAN eps/min_samples if needed.")
