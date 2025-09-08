from pathlib import Path
import json
import cv2
import numpy as np

# ================== CONFIG ==================
IN_DIR  = Path("data/data_segmented_images")   # source images
OUT_DIR = Path("decomposed")       # target root
K = 5                               # number of color clusters
MIN_AREA_FRAC = 0.005               # discard clusters < 0.5% of image pixels
BLUR = 0                            # e.g., 1 or 3 to slightly denoise before k-means
RANDOM_STATE = 42                   # reproducibility for k-means init
# ===========================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_image(path: Path):
    # Preserve alpha if present
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img

def get_rgb_alpha(img):
    """Return (rgb uint8, alpha uint8 or None) from BGR/BGRA/Gray."""
    if img is None:
        return None, None
    if img.ndim == 2:  # grayscale
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return rgb, None
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, alpha
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb, None
    return None, None

def kmeans_colors(rgb, alpha=None, k=5, blur_ksize=0, seed=42):
    """Run k-means on visible pixels; return labels (H,W), centers (k,3), valid_mask (H,W)."""
    H, W, _ = rgb.shape
    if blur_ksize and blur_ksize > 0:
        rgb_blur = cv2.GaussianBlur(rgb, (blur_ksize, blur_ksize), 0)
    else:
        rgb_blur = rgb

    if alpha is not None:
        valid_mask = alpha > 0
    else:
        # assume all pixels valid
        valid_mask = np.ones((H, W), dtype=bool)

    pts = rgb_blur[valid_mask].reshape(-1, 3).astype(np.float32)

    if pts.size == 0:
        # nothing visible
        labels_full = np.full((H, W), -1, dtype=np.int32)
        return labels_full, np.zeros((0,3), np.float32), valid_mask

    # k-means
    # OpenCV kmeans uses BGR convention agnostic since we pass raw numbers; centers are in same space as pts
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    attempts = 1
    flags = cv2.KMEANS_PP_CENTERS
    # set seed for reproducibility
    rng = np.random.default_rng(seed)
    # To stabilize init, shuffle a copy
    idx = rng.permutation(pts.shape[0])
    pts_shuf = pts[idx]

    compactness, labels, centers = cv2.kmeans(pts_shuf, k, None, criteria, attempts, flags)
    # labels are for shuffled pts; map back
    inv_idx = np.empty_like(idx)
    inv_idx[idx] = np.arange(idx.size)
    labels_unshuf = labels[inv_idx].reshape(-1)

    labels_full = np.full((H, W), -1, dtype=np.int32)
    labels_full[valid_mask] = labels_unshuf

    return labels_full, centers, valid_mask

def save_cluster_cutouts(stem_out_dir: Path, rgb, alpha_in, labels, centers, valid_mask, min_area_frac=0.005):
    H, W, _ = rgb.shape
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return []

    clusters = []
    for c in range(centers.shape[0]):
        mask = (labels == c) & valid_mask
        area = int(mask.sum())
        if area < min_area_frac * total_valid:
            continue  # skip tiny clusters

        # Build an alpha mask for this cluster
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[mask] = 255

        # Composite RGBA (keep original colors only where cluster is)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = alpha

        # Optional: tight crop to cluster bbox
        ys, xs = np.where(alpha > 0)
        if xs.size > 0:
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            rgba = rgba[y0:y1, x0:x1]

        # Save
        fname = f"color_{c:02d}.png"
        out_path = stem_out_dir / fname
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        # record cluster info
        center_rgb = centers[c].astype(float).tolist()
        clusters.append({
            "cluster_id": c,
            "file": str(out_path.name),
            "area_pixels": area,
            "area_frac": float(area / total_valid),
            "center_rgb": center_rgb  # approximate mean color for cluster
        })

    return clusters

def process_all():
    ensure_dir(OUT_DIR)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in IN_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not images:
        print(f"[warn] No images found under {IN_DIR.resolve()}")
        return

    for img_path in sorted(images):
        img = load_image(img_path)
        rgb, alpha = get_rgb_alpha(img)
        if rgb is None:
            print(f"[warn] Skipping unreadable image: {img_path}")
            continue

        stem = img_path.stem
        out_dir = OUT_DIR / stem
        ensure_dir(out_dir)

        labels, centers, valid_mask = kmeans_colors(rgb, alpha=alpha, k=K, blur_ksize=BLUR, seed=RANDOM_STATE)
        clusters = save_cluster_cutouts(out_dir, rgb, alpha, labels, centers, valid_mask, min_area_frac=MIN_AREA_FRAC)

        # Save a small manifest for this image
        manifest = {
            "source": str(img_path.relative_to(IN_DIR)),
            "k": K,
            "clusters_kept": len(clusters),
            "clusters": clusters
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[ok] {img_path.name}: {len(clusters)} clusters saved -> {out_dir}")

if __name__ == "__main__":
    process_all()
