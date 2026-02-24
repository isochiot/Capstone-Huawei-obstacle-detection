
# CADCD 3d_ann.json (3D cuboids) + calib/00.yaml + calib/extrinsics.yaml
# -> project to camera00 image_00, export YOLOv8 2D dataset
# ------------------------------------------------------------

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# =========================
# 1) PATHS (your structure)
# =========================
DRIVE_ROOT = Path(r"D:\AdverseWeather\datasets\cadcd_small")

JSON_PATH = DRIVE_ROOT / "0002" / "3d_ann.json"
CAM00_YAML = DRIVE_ROOT / "calib" / "00.yaml"
EXTR_YAML = DRIVE_ROOT / "calib" / "extrinsics.yaml"

IMAGE_DIR = DRIVE_ROOT / "0002" / "labeled" / "image_00" / "data"
OUT_DIR = DRIVE_ROOT.parent / "yolo_cadcd_0002_image00"

# =========================
# 2) SETTINGS
# =========================
TRAIN_RATIO = 0.8
MIN_BOX_PX = 4
IMG_EXTS = (".png", ".jpg", ".jpeg")

# edit if your json contains more labels
LABEL_MAP = {
    "Car": 0,
    "Truck": 1,
    "Bus": 2,
}


# ----------------------
# IO helpers
# ----------------------
def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_K_from_00_yaml(cam00: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    w = int(cam00["image_width"])
    h = int(cam00["image_height"])
    arr = cam00["camera_matrix"]["data"]
    if not (isinstance(arr, list) and len(arr) == 9):
        raise ValueError("00.yaml: camera_matrix.data must be length-9 list")
    K = np.array(arr, dtype=float).reshape(3, 3)
    return K, w, h

def parse_T_lidar_cam00(extr: Dict[str, Any]) -> np.ndarray:
    # fixed key (no guessing)
    if "T_LIDAR_CAM00" not in extr:
        raise KeyError("extrinsics.yaml missing key: T_LIDAR_CAM00")
    T = np.array(extr["T_LIDAR_CAM00"], dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T_LIDAR_CAM00 must be 4x4")
    return T


# ----------------------
# JSON field helpers (dict xyz or list xyz)
# ----------------------
def vec3(v: Any) -> Optional[np.ndarray]:
    if isinstance(v, dict) and all(k in v for k in ("x", "y", "z")):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=float)
    if isinstance(v, list) and len(v) == 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
    return None


# ----------------------
# Geometry: 3D cuboid -> corners
# ----------------------
def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def cuboid_corners(center: np.ndarray, dims: np.ndarray, yaw: float) -> np.ndarray:
    """
    center: (3,) in LiDAR frame
    dims: (3,) -> (dx,dy,dz)
    yaw: radians about z
    return: (8,3) corners in LiDAR frame
    """
    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    corners_local = np.array([[sx*hx, sy*hy, sz*hz]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    R = rotz(yaw)
    return (corners_local @ R.T) + center.reshape(1, 3)


# ----------------------
# Projection: corners -> 2D bbox
# ----------------------
def bbox_from_corners(K: np.ndarray, T_lidar_to_cam: np.ndarray, corners_lidar: np.ndarray,
                      W: int, H: int, min_box_px: int) -> Optional[Tuple[int, int, int, int]]:
    # lidar -> cam
    pts4 = np.hstack([corners_lidar, np.ones((corners_lidar.shape[0], 1), dtype=float)])  # (8,4)
    cam4 = (T_lidar_to_cam @ pts4.T).T  # (8,4)
    cam = cam4[:, :3] / cam4[:, 3:4]    # (8,3)
    Z = cam[:, 2]

    # must be in front of camera
    valid = Z > 0.05
    if valid.sum() < 4:
        return None

    camv = cam[valid]
    X, Y, Z = camv[:, 0], camv[:, 1], camv[:, 2]

    u = K[0, 0] * (X / Z) + K[0, 2]
    v = K[1, 1] * (Y / Z) + K[1, 2]

    x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())

    # clamp
    x1 = max(0.0, min(float(W - 1), x1))
    x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1))
    y2 = max(0.0, min(float(H - 1), y2))

    if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
        return None

    return int(x1), int(y1), int(x2), int(y2)

def yolo_line(cls_id: int, x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> str:
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


# ----------------------
# Auto choose T direction (T or inv(T))
# ----------------------
def score_T(frames: List[Dict[str, Any]], K: np.ndarray, T: np.ndarray, W: int, H: int, n_probe: int = 10) -> int:
    """Count how many valid boxes we can produce on first n_probe frames."""
    cnt = 0
    for i in range(min(n_probe, len(frames))):
        cuboids = frames[i].get("cuboids", [])
        if not isinstance(cuboids, list):
            continue
        for c in cuboids:
            if not isinstance(c, dict):
                continue
            lab = c.get("label")
            if lab not in LABEL_MAP:
                continue
            center = vec3(c.get("position"))
            dims = vec3(c.get("dimensions"))
            if center is None or dims is None:
                continue
            yaw = float(c.get("yaw", 0.0))
            corners = cuboid_corners(center, dims, yaw)
            bb = bbox_from_corners(K, T, corners, W, H, MIN_BOX_PX)
            if bb is not None:
                cnt += 1
    return cnt


def main() -> None:
    # sanity checks
    for p in [JSON_PATH, CAM00_YAML, EXTR_YAML]:
        if not p.exists():
            raise FileNotFoundError(str(p))
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(str(IMAGE_DIR))

    frames = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise ValueError("3d_ann.json must be a non-empty list")
    if not isinstance(frames[0], dict) or "cuboids" not in frames[0]:
        raise ValueError("Each frame must be a dict containing key 'cuboids'")

    imgs = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not imgs:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    # read actual image size
    im0 = cv2.imread(str(imgs[0]))
    if im0 is None:
        raise RuntimeError(f"Cannot read image: {imgs[0]}")
    H, W = im0.shape[:2]

    # calib
    cam00 = load_yaml(CAM00_YAML)
    K, calib_w, calib_h = parse_K_from_00_yaml(cam00)
    extr = load_yaml(EXTR_YAML)
    T_raw = parse_T_lidar_cam00(extr)
    T_inv = np.linalg.inv(T_raw)

    if (W, H) != (calib_w, calib_h):
        print(f"[WARN] Image size ({W}x{H}) != calib size ({calib_w}x{calib_h}). Using image size.")

    # choose correct direction automatically
    s_raw = score_T(frames, K, T_raw, W, H, n_probe=10)
    s_inv = score_T(frames, K, T_inv, W, H, n_probe=10)
    if s_inv > s_raw:
        T = T_inv
        print(f"[INFO] Using inv(T_LIDAR_CAM00) (score inv={s_inv} > raw={s_raw})")
    else:
        T = T_raw
        print(f"[INFO] Using T_LIDAR_CAM00 as-is (score raw={s_raw} >= inv={s_inv})")

    # alignment
    n_json = len(frames)
    n_img = len(imgs)
    n = min(n_json, n_img)
    if n_json != n_img:
        print(f"[WARN] json frames={n_json} but images={n_img}. Using first min={n} by sorted order alignment.")

    # output dirs
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = int(n * TRAIN_RATIO)
    kept = 0
    dropped_no_box = 0

    for i in tqdm(range(n), desc="3D->2D->YOLO (cam00)"):
        split = "train" if i < n_train else "val"
        cuboids = frames[i].get("cuboids", [])
        if not isinstance(cuboids, list) or not cuboids:
            continue

        lines: List[str] = []

        for c in cuboids:
            if not isinstance(c, dict):
                continue

            lab = c.get("label")
            if lab not in LABEL_MAP:
                continue

            center = vec3(c.get("position"))
            dims = vec3(c.get("dimensions"))
            if center is None or dims is None:
                continue

            yaw = float(c.get("yaw", 0.0))
            corners = cuboid_corners(center, dims, yaw)
            bb = bbox_from_corners(K, T, corners, W, H, MIN_BOX_PX)
            if bb is None:
                continue

            x1, y1, x2, y2 = bb
            lines.append(yolo_line(LABEL_MAP[lab], x1, y1, x2, y2, W, H))

        if not lines:
            dropped_no_box += 1
            continue

        src_img = imgs[i]
        dst_img = OUT_DIR / "images" / split / src_img.name
        dst_lbl = OUT_DIR / "labels" / split / (src_img.stem + ".txt")

        shutil.copy2(src_img, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        kept += 1

    # data.yaml
    names = {v: k for k, v in LABEL_MAP.items()}
    data_yaml = {
        "path": str(OUT_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (OUT_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Kept labeled images: {kept}")
    print(f"Dropped (no 2D boxes after projection): {dropped_no_box}")
    print(f"YOLO dataset at: {OUT_DIR}")
    print("\nTrain example:")
    print(f"  yolo detect train data={OUT_DIR}\\data.yaml model=yolov8n.pt imgsz=640 epochs=20 batch=8")


if __name__ == "__main__":
    main()