from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
import json
import re

# ====== CONFIG ======
DATA_DIR = Path("data/data_objects/images")   # path to your COIL-100 images
OUT_DIR = Path("coil_out")                    # where results will go
MAKE_MASKS = True
MAKE_CUTOUTS = True
TIGHT_CROP = True
CARDINAL_ONLY = True   # only 000,090,180,270 views
# ====================

FNAME_RE = re.compile(r"obj(\d{1,3})__(\d{3})\.(png|jpg|jpeg|ppm)$", re.IGNORECASE)

def list_images(data_dir: Path):
    return [p for p in sorted(data_dir.rglob("*")) if p.is_file() and FNAME_RE.match(p.name)]

def parse_id_angle(name: str):
    m = FNAME_RE.match(name)
    if not m: return None, None
    return int(m.group(1)), int(m.group(2))

def summarize(files):
    per_obj = defaultdict(list)
    for f in files:
        oid, ang = parse_id_angle(f.name)
        if oid is not None: per_obj[oid].append((ang, f))
    for oid in per_obj: per_obj[oid].sort(key=lambda x: x[0])
    shots_counts = {oid: len(v) for oid, v in per_obj.items()}
    counts = list(shots_counts.values())
    stats = {
        "num_objects": len(per_obj),
        "shots_per_object_min": int(min(counts)) if counts else 0,
        "shots_per_object_max": int(max(counts)) if counts else 0,
        "shots_per_object_mean": float(np.mean(counts)) if counts else 0.0,
    }
    return per_obj, shots_counts, stats

def largest_component_mask(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype("uint8")

def make_object_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return largest_component_mask(mask)

def apply_alpha_cutout(bgr, mask):
    b, g, r = cv2.split(bgr)
    return cv2.merge([b, g, r, mask])

def tight_crop(img, mask, pad=2):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return img, mask
    x0, x1 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad)
    y0, y1 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad)
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

def main():
    files = list_images(DATA_DIR)
    per_obj, shots_counts, stats = summarize(files)

    print("COIL-100 SUMMARY")
    print(stats)
    from collections import Counter
    print("Shots per object distribution:", dict(Counter(shots_counts.values())))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"stats": stats, "shots_per_object": shots_counts}, f, indent=2)

    if MAKE_MASKS or MAKE_CUTOUTS:
        mask_dir = OUT_DIR / "masks"
        cut_dir  = OUT_DIR / "cutouts"
        if MAKE_MASKS:
            mask_dir.mkdir(parents=True, exist_ok=True)   # <-- create folder
        if MAKE_CUTOUTS:
            cut_dir.mkdir(parents=True, exist_ok=True)    # <-- create folder

        cardinal = {0, 90, 180, 270}
        n_ok, n_fail = 0, 0

        for oid, angle_files in per_obj.items():
            for ang, path in angle_files:
                if CARDINAL_ONLY and ang not in cardinal:
                    continue

                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"[warn] could not read {path}")
                    n_fail += 1
                    continue

                mask = make_object_mask(bgr)
                if TIGHT_CROP:
                    bgr, mask = tight_crop(bgr, mask, pad=2)

                rel = f"obj{oid:03d}__{ang:03d}.png"

                if MAKE_MASKS:
                    ok = cv2.imwrite(str(mask_dir / rel), mask)
                    if not ok:
                        print(f"[warn] failed to write mask {mask_dir/rel}")
                        n_fail += 1

                if MAKE_CUTOUTS:
                    cut = apply_alpha_cutout(bgr, mask)  # BGRA (alpha from mask)
                    ok = cv2.imwrite(str(cut_dir / rel), cut)
                    if not ok:
                        print(f"[warn] failed to write cutout {cut_dir/rel}")
                        n_fail += 1
                    else:
                        n_ok += 1

        print(f"Done. Wrote {n_ok} files, {n_fail} warnings.")
        if MAKE_CUTOUTS:
            print(f"Transparent cutouts: {cut_dir}")
        if MAKE_MASKS:
            print(f"Masks: {mask_dir}")

if __name__ == "__main__":
    main()
