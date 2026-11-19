import argparse, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, resolve_device
from src.pose_utils import COCO_SKELETON

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--model", default="yolo11n-pose.pt")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--no-rotate", action="store_true")
p.add_argument("--kp-conf", type=float, default=0.5)
a = p.parse_args()

cfg = load_config(); device = resolve_device(cfg["device"])
model = YOLO(a.model)
src = Path(a.source)
cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ok, frame = cap.read(); H, W = frame.shape[:2]
tag = f"{Path(a.model).stem.replace('-pose','')}_{'norot' if a.no_rotate else 'rot'}"
out = ROOT / "reports" / "day02" / "videos" / f"{src.stem}_{tag}_s{a.imgsz}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

ROTS = {"0": None} if a.no_rotate else {"0": None, "90": cv2.ROTATE_90_CLOCKWISE, "270": cv2.ROTATE_90_COUNTERCLOCKWISE}

def back(xy, rot):
    # 회전된 이미지 좌표 → 원본 좌표
    if rot == "90":  return np.stack([xy[:, 1], H - 1 - xy[:, 0]], 1)
    if rot == "270": return np.stack([W - 1 - xy[:, 1], xy[:, 0]], 1)
    return xy

picked = {k: 0 for k in ROTS}; confs = []; miss = n = 0
while ok:
    n += 1
    best = None
    for k, r in ROTS.items():
        img = frame if r is None else cv2.rotate(frame, r)
        res = model.predict(img, imgsz=a.imgsz, conf=cfg["person_confidence"], max_det=1,
                            device=device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints.conf is None:
            continue
        c = res.keypoints.conf.cpu().numpy()[0]
        if best is None or c.mean() > best[2].mean():
            best = (k, res.keypoints.xy.cpu().numpy()[0], c)

    vis = frame.copy()
    if best is None:
        miss += 1
        cv2.putText(vis, "NO PERSON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        k, xy, c = best; xy = back(xy, k); picked[k] += 1; confs.append(c)
        for s, e in COCO_SKELETON:
            if c[s] >= a.kp_conf and c[e] >= a.kp_conf:
                cv2.line(vis, tuple(xy[s].astype(int)), tuple(xy[e].astype(int)), (0, 255, 0), 3)
        for i, (pt, cf) in enumerate(zip(xy, c)):
            if cf >= a.kp_conf:
                cv2.circle(vis, tuple(pt.astype(int)), 6, (0, 255, 255), -1)
        cv2.putText(vis, f"rot={k} mean={c.mean():.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    wr.write(vis); ok, frame = cap.read()

cap.release(); wr.release()
c = np.array(confs) if confs else np.zeros((1, 17))
print(f"{tag:12s} | 평균 {c.mean():.3f} | 손목 {c[:, [9,10]].mean():.3f} | 발목 {c[:, [15,16]].mean():.3f} | "
      f"LOW {(c < a.kp_conf).mean():.1%} | 미검출 {miss}/{n} | 선택 방향 {picked}")
print(f"Saved: {out}")
