import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, resolve_device
from src.pose_utils import COCO_SKELETON

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--model", default="yolo11s-pose.pt")
p.add_argument("--imgsz", type=int, default=960)
p.add_argument("--win", type=int, default=15, help="회전 방향 다수결 Frame 수")
p.add_argument("--smooth", type=int, default=1, help="이동평균 Frame 수 (1 = 끔)")
p.add_argument("--interp", type=int, default=4, help="최대 보간 Frame 수 (0 = 끔)")
p.add_argument("--swap", action="store_true", help="좌우 뒤바뀜 보정")
p.add_argument("--kp-conf", type=float, default=0.5)
p.add_argument("--rots", default="0,90,270", help="쓸 회전 방향")
a = p.parse_args()

cfg = load_config(); device = resolve_device(cfg["device"]); model = YOLO(a.model)
src = Path(a.source); BODY = list(range(5, 17))
ROTS = {"0": None, "90": cv2.ROTATE_90_CLOCKWISE, "270": cv2.ROTATE_90_COUNTERCLOCKWISE}
ROTS = {k: v for k, v in ROTS.items() if k in a.rots.split(",")}
GROUPS = {"다리": [(11, 12), (13, 14), (15, 16)], "팔": [(5, 6), (7, 8), (9, 10)]}

# 1차: 세 방향 추론
cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ok, frame = cap.read(); H, W = frame.shape[:2]
def back(xy, k):
    if k == "90":  return np.stack([xy[:, 1], H - 1 - xy[:, 0]], 1)
    if k == "270": return np.stack([W - 1 - xy[:, 1], xy[:, 0]], 1)
    return xy
cand, n = [], 0
while ok:
    n += 1; row = {}
    for k, r in ROTS.items():
        img = frame if r is None else cv2.rotate(frame, r)
        res = model.predict(img, imgsz=a.imgsz, conf=cfg["person_confidence"], max_det=1,
                            device=device, verbose=False)[0]
        if len(res.boxes) and res.keypoints.conf is not None:
            row[k] = (back(res.keypoints.xy.cpu().numpy()[0], k), res.keypoints.conf.cpu().numpy()[0])
    cand.append(row)
    if n % 300 == 0: print(f"추론 {n} Frame")
    ok, frame = cap.read()
cap.release()

# 방향: 몸 관절 점수 + 다수결
keys = list(ROTS)
score = np.array([[c[k][1][BODY].mean() if k in c else 0.0 for k in keys] for c in cand])
best = score.argmax(1); half = a.win // 2
chosen = [keys[np.bincount(best[max(0, i - half): i + half + 1], minlength=3).argmax()] for i in range(n)]
XY = np.full((n, 17, 2), np.nan); CF = np.zeros((n, 17))
for i, (c, k) in enumerate(zip(cand, chosen)):
    if k not in c: k = keys[best[i]] if keys[best[i]] in c else None
    if k: XY[i], CF[i] = c[k]
raw_low = (CF < a.kp_conf).mean()

# 좌우 보정: 직전 Frame과 비교해 묶음 단위로 교체
swaps = {g: 0 for g in GROUPS}
if a.swap:
    last = XY[0].copy()
    for i in range(1, n):
        for g, pairs in GROUPS.items():
            L = [l for l, _ in pairs]; R = [r for _, r in pairs]
            ok_ = (CF[i, L] >= 0.3) & (CF[i, R] >= 0.3) & ~np.isnan(last[L, 0]) & ~np.isnan(last[R, 0])
            if ok_.sum() == 0: continue
            Lv, Rv = np.array(L)[ok_], np.array(R)[ok_]
            same = np.linalg.norm(XY[i, Lv] - last[Lv], axis=1).sum() + np.linalg.norm(XY[i, Rv] - last[Rv], axis=1).sum()
            cross = np.linalg.norm(XY[i, Lv] - last[Rv], axis=1).sum() + np.linalg.norm(XY[i, Rv] - last[Lv], axis=1).sum()
            if cross < same * 0.7:
                XY[i, L + R] = XY[i, R + L]; CF[i, L + R] = CF[i, R + L]; swaps[g] += 1
        good = CF[i] >= 0.3
        last[good] = XY[i, good]

# 보간 + 이동평균
for j in range(17):
    for d in range(2):
        s = pd.Series(np.where(CF[:, j] >= 0.3, XY[:, j, d], np.nan))
        if a.interp > 0: s = s.interpolate(limit=a.interp, limit_area="inside")
        if a.smooth > 1: s = s.rolling(a.smooth, center=True, min_periods=1).mean()
        XY[:, j, d] = s.values
filled = ~np.isnan(XY[:, :, 0])

# 2차: 그리기
tag = f"{Path(a.model).stem}_sm{a.smooth}_ip{a.interp}{'_swap' if a.swap else ''}_r{a.rots.replace(",", "-")}_s{a.imgsz}"
out = ROOT / "reports" / "day02" / "videos" / f"{src.stem}_{tag}.mp4"
wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
cap = cv2.VideoCapture(str(src)); i = 0; ok, frame = cap.read()
while ok and i < n:
    for s_, e_ in COCO_SKELETON:
        if filled[i, s_] and filled[i, e_]:
            cv2.line(frame, tuple(XY[i, s_].astype(int)), tuple(XY[i, e_].astype(int)), (0, 255, 0), 3)
    for j in range(17):
        if filled[i, j]:
            color = (0, 255, 255) if j % 2 == 1 else (255, 0, 255)   # 노랑 = left, 보라 = right
            cv2.circle(frame, tuple(XY[i, j].astype(int)), 6, color, -1)
    cv2.putText(frame, f"frame={i+1} rot={chosen[i]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    wr.write(frame); ok, frame = cap.read(); i += 1
cap.release(); wr.release()

flips = sum(chosen[k] != chosen[k - 1] for k in range(1, n))
print(f"{tag} | 원본 LOW {raw_low:.1%} → 빈 관절 {(~filled).mean():.1%} | 방향 전환 {flips}회 | 좌우 교체 {swaps}")
print(f"Saved: {out}")
