# file: color_decompose.py  (overwrite your previous one)

from pathlib import Path
import json, cv2, numpy as np

IN_DIR  = Path("data/data_segmented_images")   # <-- updated path
OUT_DIR = Path("decomposed")
K = 5
MIN_AREA_FRAC = 0.005
BLUR = 0
RANDOM_STATE = 42

def ensure_dir(p): p.mkdir(parents=True, exist_ok=True); return p
def load_image(p): return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

def get_rgb_alpha(img):
    if img is None: return None, None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), None
    if img.shape[2] == 4:
        rgb = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB); alpha = img[:,:,3]; return rgb, alpha
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
    return None, None

def kmeans_colors(rgb, alpha=None, k=5, blur_ksize=0, seed=42):
    H,W,_ = rgb.shape
    rgb_blur = cv2.GaussianBlur(rgb,(blur_ksize,blur_ksize),0) if blur_ksize else rgb
    valid_mask = (alpha>0) if alpha is not None else np.ones((H,W), bool)
    pts = rgb_blur[valid_mask].reshape(-1,3).astype(np.float32)
    if pts.size == 0:
        return np.full((H,W), -1, np.int32), np.zeros((0,3), np.float32), valid_mask
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,0.2)
    flags=cv2.KMEANS_PP_CENTERS
    rng = np.random.default_rng(seed); idx = rng.permutation(len(pts)); pts_shuf = pts[idx]
    _, labels, centers = cv2.kmeans(pts_shuf, k, None, criteria, 1, flags)
    inv = np.empty_like(idx); inv[idx]=np.arange(idx.size); labels_unshuf = labels[inv].reshape(-1)
    labels_full = np.full((H,W), -1, np.int32); labels_full[valid_mask]=labels_unshuf
    return labels_full, centers, valid_mask

def save_cluster_cutouts(stem_out_dir, rgb, labels, valid_mask, min_area_frac):
    H,W,_ = rgb.shape
    total_valid = int(valid_mask.sum()); clusters=[]
    for c in np.unique(labels[valid_mask]):
        if c < 0: continue
        mask = (labels==c) & valid_mask
        area = int(mask.sum())
        if area < min_area_frac*total_valid: continue
        alpha = np.zeros((H,W), np.uint8); alpha[mask]=255
        ys, xs = np.where(alpha>0)
        y0,y1 = ys.min(), ys.max()+1; x0,x1 = xs.min(), xs.max()+1
        rgba = np.zeros((y1-y0, x1-x0, 4), np.uint8)
        rgba[:,:,:3] = rgb[y0:y1, x0:x1]
        rgba[:,:,3]  = alpha[y0:y1, x0:x1]
        out = stem_out_dir / f"color_{c:02d}.png"
        ensure_dir(out.parent)
        cv2.imwrite(str(out), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        clusters.append({
            "cluster_id": int(c),
            "file": out.name,
            "area_pixels": area,
            "bbox": {"x": int(x0), "y": int(y0), "w": int(x1-x0), "h": int(y1-y0)}
        })
    return clusters

def process_all():
    ensure_dir(OUT_DIR)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    images=[p for p in IN_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not images:
        print(f"[warn] No images in {IN_DIR}"); return
    for img_path in sorted(images):
        img = load_image(img_path); rgb, alpha = get_rgb_alpha(img)
        if rgb is None: print("[warn] unreadable", img_path); continue
        H,W = rgb.shape[:2]
        stem = img_path.stem
        out_dir = OUT_DIR / stem
        labels, centers, valid_mask = kmeans_colors(rgb, alpha, k=K, blur_ksize=BLUR, seed=RANDOM_STATE)
        clusters = save_cluster_cutouts(out_dir, rgb, labels, valid_mask, MIN_AREA_FRAC)
        manifest = {
            "source": str(img_path.relative_to(IN_DIR)),
            "orig_w": int(W), "orig_h": int(H),
            "k": int(K),
            "parts": clusters
        }
        with open(out_dir/"manifest.json","w") as f: json.dump(manifest, f, indent=2)
        print(f"[ok] {img_path.name}: {len(clusters)} parts -> {out_dir}")

if __name__=="__main__":
    process_all()
