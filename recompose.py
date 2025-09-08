from pathlib import Path
import csv
import json
import cv2
import numpy as np

# ========= CONFIG =========
DECOMP_DIR     = Path("decomposed")                  # where color_XX.png live (per image subfolders)
MATCHES_CSV    = DECOMP_DIR / "matches_to_coil.csv"  # created earlier
SEGMENTED_DIR  = Path("data/data_segmented_images")              # original segmented images (e.g., img1.png, img2.png)
REPLACEMENTS   = Path("replacements")                # per-part resized COIL cutouts
RECONSTRUCTED  = Path("reconstructed")               # per-image recomposed canvases
COIL_ROOT      = Path("coil_out/cutouts")            # base for COIL paths in CSV (for sanity checks)
# ==========================

# ---------- utils ----------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_rgba(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:  # gray -> BGRA
        a = np.full(img.shape, 255, np.uint8)
        img = cv2.merge([img, img, img, a])
    elif img.shape[2] == 3:  # BGR -> BGRA
        h, w = img.shape[:2]
        a = np.full((h, w), 255, np.uint8)
        img = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], a])
    return img

def alpha_mask(rgba):
    a = rgba[:, :, 3]
    m = (a > 0).astype(np.uint8) * 255
    # keep largest component (robustness)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    lab = 1 + int(np.argmax(areas))
    return np.where(lbl == lab, 255, 0).astype(np.uint8)

def bbox_from_mask(mask, pad=0):
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    if pad:
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(mask.shape[1], x1 + pad); y1 = min(mask.shape[0], y1 + pad)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

def resize_rgba(rgba, width, height):
    return cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA)

def paste_rgba(dst_rgba, src_rgba, x, y):
    """Alpha composite src onto dst at (x,y). Both BGRA uint8."""
    H, W = dst_rgba.shape[:2]
    h, w = src_rgba.shape[:2]
    if w <= 0 or h <= 0:
        return

    # Clip to canvas bounds
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, x0 + int(w)), min(H, y0 + int(h))
    if x1 <= x0 or y1 <= y0:
        return

    # Corresponding src ROI
    sx0, sy0 = x0 - int(x), y0 - int(y)
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

    dst_roi = dst_rgba[y0:y1, x0:x1].astype(np.float32)
    src_roi = src_rgba[sy0:sy1, sx0:sx1].astype(np.float32)

    alpha = (src_roi[:, :, 3:4] / 255.0)
    out_rgb = alpha * src_roi[:, :, :3] + (1.0 - alpha) * dst_roi[:, :, :3]
    out_a   = np.maximum(dst_roi[:, :, 3:4], src_roi[:, :, 3:4])

    dst_rgba[y0:y1, x0:x1, :3] = out_rgb.clip(0, 255).astype(np.uint8)
    dst_rgba[y0:y1, x0:x1, 3:4] = out_a.clip(0, 255).astype(np.uint8)

def find_original_image(group_stem: str):
    """Find original segmented image for this group by trying common extensions."""
    for ext in IMG_EXTS:
        cand = SEGMENTED_DIR / f"{group_stem}{ext}"
        if cand.exists():
            return cand
    # also search recursively just in case
    for p in SEGMENTED_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == group_stem:
            return p
    return None

def get_bbox_from_manifest(group_stem: str, part_stem: str):
    """Read bbox (x,y,w,h) if present in decomposed/<group>/manifest.json."""
    mpath = DECOMP_DIR / group_stem / "manifest.json"
    if not mpath.exists():
        return None, None
    try:
        data = json.loads(Path(mpath).read_text())
    except Exception:
        return None, None
    orig_size = None
    if "orig_w" in data and "orig_h" in data:
        orig_size = (int(data["orig_w"]), int(data["orig_h"]))
    # find entry for this part
    if "parts" in data:
        for p in data["parts"]:
            if p.get("file", "").startswith(part_stem):
                b = p.get("bbox")
                if b and all(k in b for k in ("x","y","w","h")):
                    return (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])), orig_size
    # older manifest format (clusters list)
    if "clusters" in data:  # try alternate key names if you previously used 'clusters'
        for p in data["clusters"]:
            if p.get("file","").startswith(part_stem):
                b = p.get("bbox")
                if b and all(k in b for k in ("x","y","w","h")):
                    return (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])), orig_size
    return None, orig_size

def infer_bbox_by_template(group_stem: str, part_rgba):
    """
    Fallback: locate part on original image using alpha-template matching on masks.
    """
    orig_path = find_original_image(group_stem)
    if orig_path is None:
        return None, None
    orig_rgba = load_rgba(orig_path)
    if orig_rgba is None:
        return None, None

    orig_mask = (orig_rgba[:, :, 3] > 0).astype(np.uint8) * 255
    part_m = alpha_mask(part_rgba)

    # use matchTemplate on float masks
    OM = orig_mask.astype(np.float32) / 255.0
    PM = part_m.astype(np.float32) / 255.0
    oh, ow = OM.shape
    ph, pw = PM.shape
    if ph >= oh or pw >= ow:
        # If the part is same size as original, assume full-frame
        return (0, 0, ow, oh), (ow, oh)

    res = cv2.matchTemplate(OM, PM, cv2.TM_CCORR_NORMED)
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    x, y = maxloc
    return (int(x), int(y), int(pw), int(ph)), (ow, oh)

# ---------- main ----------
def run():
    ensure_dir(REPLACEMENTS)
    ensure_dir(RECONSTRUCTED)

    # Load matches
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {MATCHES_CSV}")
    rows = []
    with open(MATCHES_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[warn] No rows in matches CSV.")
        return

    # Group by image stem
    groups = {}
    for row in rows:
        g = row["decomp_group"]
        groups.setdefault(g, []).append(row)

    for g, items in groups.items():
        # Prepare canvas
        # Try manifest for original size
        any_manifest = DECOMP_DIR / g / "manifest.json"
        orig_w = orig_h = None
        if any_manifest.exists():
            try:
                md = json.loads(any_manifest.read_text())
                if "orig_w" in md and "orig_h" in md:
                    orig_w, orig_h = int(md["orig_w"]), int(md["orig_h"])
            except Exception:
                pass

        if orig_w is None or orig_h is None:
            # Read original segmented image
            orig_path = find_original_image(g)
            if orig_path is not None:
                oimg = load_rgba(orig_path)
                if oimg is not None:
                    orig_h, orig_w = oimg.shape[:2]

        if orig_w is None or orig_h is None:
            print(f"[warn] Could not determine original size for {g}; skipping.")
            continue

        canvas = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)

        # Process parts
        for row in items:
            part_path = Path(row["decomp_path"])
            coil_path = Path(row["best_coil_path"])

            # Resolve relative-to-project paths if needed
            if not part_path.is_absolute():
                part_path = Path(part_path)
            if not coil_path.is_absolute():
                coil_path = coil_path if coil_path.is_file() else (COIL_ROOT / Path(coil_path).name)

            part_rgba = load_rgba(part_path)
            coil_rgba = load_rgba(coil_path)
            if part_rgba is None or coil_rgba is None:
                print(f"[skip] missing image for {g}: {part_path} or {coil_path}")
                continue

            # Get bbox from manifest; else infer
            bbox, _sz = get_bbox_from_manifest(g, Path(part_path).stem)
            if bbox is None:
                bbox, _sz = infer_bbox_by_template(g, part_rgba)
            if bbox is None:
                # last resort: drop at 0,0 with its own size
                bmask = alpha_mask(part_rgba)
                bb = bbox_from_mask(bmask)
                if bb is None:
                    print(f"[skip] no bbox for {part_path}")
                    continue
                x, y, w, h = 0, 0, bb[2], bb[3]
            else:
                x, y, w, h = bbox

            # Resize matched COIL cutout to bbox size and save per-part replacement
            coil_resized = resize_rgba(coil_rgba, w, h)
            rep_dir = ensure_dir(REPLACEMENTS / g)
            cv2.imwrite(str(rep_dir / f"{Path(part_path).stem}_replacement.png"), coil_resized)

            # Paste onto canvas
            paste_rgba(canvas, coil_resized, x, y)

        # Save recomposed image
        out_img = RECONSTRUCTED / f"{g}.png"
        cv2.imwrite(str(out_img), canvas)
        print(f"[ok] Reconstructed {g} -> {out_img}")

if __name__ == "__main__":
    run()
