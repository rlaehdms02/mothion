import argparse, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, resolve_device

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--imgsz", type=int, required=True)
args = p.parse_args()

src = Path(args.source)
cfg = load_config()
model = YOLO(cfg["pose_model"])
device = resolve_device(cfg["device"])

cap = cv2.VideoCapture(str(src))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ok, frame = cap.read()
h, w = frame.shape[:2]
out = ROOT / "reports" / "day02" / "videos" / f"{src.stem}_imgsz{args.imgsz}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

confs, miss, n = [], 0, 0
while ok:
    n += 1
    r = model.predict(frame, imgsz=args.imgsz, conf=cfg["person_confidence"],
                      device=device, verbose=False)[0]
    writer.write(r.plot())
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints.conf is None:
        miss += 1
    else:
        confs.append(r.keypoints.conf.cpu().numpy()[int(r.boxes.conf.argmax())])
    ok, frame = cap.read()
cap.release(); writer.release()

c = np.array(confs) if confs else np.zeros((1, 17))
print(f"imgsz={args.imgsz:4d} | 평균 {c.mean():.3f} | 팔꿈치 {c[:, [7, 8]].mean():.3f} | "
      f"손목 {c[:, [9, 10]].mean():.3f} | 무릎 {c[:, [13, 14]].mean():.3f} | "
      f"발목 {c[:, [15, 16]].mean():.3f} | LOW(<0.5) 비율 {(c < 0.5).mean():.1%} | 미검출 {miss}/{n}")
print(f"Saved: {out}")
