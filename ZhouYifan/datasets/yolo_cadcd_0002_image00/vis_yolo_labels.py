import random
from pathlib import Path
import cv2
from tqdm import tqdm

# ===== 你需要改这 3 个 =====
YOLO_ROOT = Path(r"D:\AdverseWeather\datasets\yolo_cadcd_0002_image00")
SPLIT = "train"   # "train" 或 "val"
N_SAMPLES = 30    # 抽多少张可视化

# 如果你的 data.yaml 里 names 是 {0:Car,1:Truck...}，这里会自动解析
DATA_YAML = YOLO_ROOT / "data.yaml"

def load_names():
    import yaml
    d = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = d.get("names", {})
    # names 可能是 dict[int->str] 或 list[str]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            out[int(k)] = str(v)
        return out
    return {}

def yolo_to_xyxy(line, W, H):
    # line: "cls xc yc w h" 归一化
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:])
    x1 = (xc - bw / 2) * W
    y1 = (yc - bh / 2) * H
    x2 = (xc + bw / 2) * W
    y2 = (yc + bh / 2) * H
    # clamp
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return cls, int(x1), int(y1), int(x2), int(y2)

def main():
    names = load_names()

    img_dir = YOLO_ROOT / "images" / SPLIT
    lbl_dir = YOLO_ROOT / "labels" / SPLIT
    out_dir = YOLO_ROOT / "vis" / SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")

    samples = imgs if len(imgs) <= N_SAMPLES else random.sample(imgs, N_SAMPLES)

    for img_path in tqdm(samples, desc="Visualizing"):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        H, W = im.shape[:2]
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if lbl_path.exists():
            lines = [ln for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for ln in lines:
                r = yolo_to_xyxy(ln, W, H)
                if r is None:
                    continue
                cls, x1, y1, x2, y2 = r
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = names.get(cls, str(cls))
                cv2.putText(im, tag, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / img_path.name), im)

    print(f"Saved visualizations to: {out_dir}")

if __name__ == "__main__":
    main()