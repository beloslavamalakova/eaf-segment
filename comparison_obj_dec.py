from pathlib import Path
import csv
import math
import cv2
import numpy as np

# ============ CONFIG ============
DECOMPOSED_DIR = Path("decomposed")         # output from your color_decompose script
COIL_CUTS_DIR  = Path("coil_out/cutouts")   # transparent COIL cutouts (BGRA PNGs)
OUT_CSV        = DECOMPOSED_DIR / "matches_to_coil.csv"

# Weights for the combined score (lower = better)
W_HU   = 1.0       # weight for Hu-moment L2 distance (shape)
W_SIZE = 0.5       # weight for size penalty (relative area difference)
MIN_VISIBLE_PIXELS = 25  # skip tiny fragments
# =================================

def load_rgba(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Ensure 4-channel
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        bgr = img
        a = np.full((img.shape[0], img.shape[1]), 255, np.uint8)
        img = cv2.merge([bgr[:,:,0], bgr[:,:,1], bgr[:,:,2], a])
    return img

def mask_from_alpha(rgba):
    alpha = rgba[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    # keep largest component to remove stray pixels
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)
    return np.where(lbl == largest, 255, 0).astype(np.uint8)

def shape_features(mask):
    """Area + log-Hu moments (scale/rotation invariant)."""
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area < MIN_VISIBLE_PIXELS:
        return area, np.zeros(7, dtype=float)
    moms = cv2.moments(m)
    hu = cv2.HuMoments(moms).flatten()
    # log transform for stability; preserve sign
    hu_log = np.array([(-np.sign(h) * np.log10(abs(h))) if h != 0 else 0.0 for h in hu], dtype=float)
    return area, hu_log

def l2(a, b):
    return float(np.linalg.norm(a - b))

def size_penalty(area_a, area_b):
    if area_a <= 0 or area_b <= 0:
        return 1.0
    # relative difference in log-area (scale-insensitive but rewards similar sizes)
    return abs(math.log(area_a) - math.log(area_b))

def combined_score(hu_a, area_a, hu_b, area_b, w_hu=W_HU, w_size=W_SIZE):
    return w_hu * l2(hu_a, hu_b) + w_size * size_penalty(area_a, area_b)

def collect_decomposed_parts(root: Path):
    """Return list of (path, stem_group, part_name)."""
    parts = []
    for img_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(img_dir.glob("color_*.png")):
            parts.append((p, img_dir.name, p.stem))  # e.g., ("decomposed/img42/color_00.png","img42","color_00")
    return parts

def collect_coil_cutouts(root: Path):
    """Return list of (path, obj_id, angle)."""
    items = []
    for p in sorted(root.rglob("*.png")):
        # try parse obj id and angle (non-fatal if missing)
        obj_id, ang = None, None
        name = p.name
        # obj123__045.png
        try:
            if "obj" in name:
                s = name.lower()
                i = s.find("obj")
                if i >= 0:
                    obj_id = int(s[i+3:i+6])
                j = s.find("__")
                if j >= 0:
                    ang = int(s[j+2:j+5])
        except Exception:
            pass
        items.append((p, obj_id, ang))
    return items

def main():
    # Load COIL features once
    coil_items = collect_coil_cutouts(COIL_CUTS_DIR)
    coil_feats = []
    for path, obj_id, ang in coil_items:
        rgba = load_rgba(path)
        if rgba is None:
            continue
        mask = mask_from_alpha(rgba)
        area, hu = shape_features(mask)
        if area < MIN_VISIBLE_PIXELS:
            continue
        coil_feats.append({
            "path": path,
            "obj_id": obj_id,
            "angle": ang,
            "area": area,
            "hu": hu
        })
    if not coil_feats:
        print(f"[error] No valid COIL cutouts found in {COIL_CUTS_DIR.resolve()}")
        return

    # Go through decomposed parts and match
    parts = collect_decomposed_parts(DECOMPOSED_DIR)
    if not parts:
        print(f"[error] No decomposed parts found in {DECOMPOSED_DIR.resolve()}")
        return

    rows = []
    for part_path, group_stem, part_name in parts:
        rgba = load_rgba(part_path)
        if rgba is None:
            continue
        mask = mask_from_alpha(rgba)
        area_p, hu_p = shape_features(mask)
        if area_p < MIN_VISIBLE_PIXELS:
            continue

        # find best COIL by combined score
        best = None
        best_score = float("inf")
        best_hu = None
        for cf in coil_feats:
            sc = combined_score(hu_p, area_p, cf["hu"], cf["area"])
            if sc < best_score:
                best_score = sc
                best = cf
                # cache distance pieces too
                best_hu = l2(hu_p, cf["hu"])
        if best is None:
            continue

        rows.append({
            "decomp_group": group_stem,
            "decomp_part": part_name,
            "decomp_path": str(part_path),
            "decomp_area": area_p,
            "best_coil_obj": best["obj_id"] if best["obj_id"] is not None else "",
            "best_coil_angle": best["angle"] if best["angle"] is not None else "",
            "best_coil_path": str(best["path"]),
            "hu_distance": round(best_hu, 6),
            "size_penalty": round(size_penalty(area_p, best["area"]), 6),
            "combined_score": round(best_score, 6),
        })

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = [
            "decomp_group","decomp_part","decomp_path","decomp_area",
            "best_coil_obj","best_coil_angle","best_coil_path",
            "hu_distance","size_penalty","combined_score"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] {len(rows)} matches written to {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
