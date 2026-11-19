import sys
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

src = Path(sys.argv[1])
out = Path("reports/day02/videos") / f"{src.stem}_mediapipe.mp4"
out.parent.mkdir(parents=True, exist_ok=True)

opts = vision.PoseLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path="models/pose_landmarker_heavy.task"),
    running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3)
lm = vision.PoseLandmarker.create_from_options(opts)

# 몸 연결선 (MediaPipe 33 번호)
LINES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
         (23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]
LEFT = {11,13,15,23,25,27,29,31}

cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ok, f = cap.read()
if not ok: raise RuntimeError(f"영상을 열 수 없습니다: {src}")
H, W = f.shape[:2]
wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
n = miss = 0
while ok:
    n += 1
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    r = lm.detect_for_video(img, int((n - 1) * 1000 / fps))
    if r.pose_landmarks:
        pts = [(int(p.x * W), int(p.y * H)) for p in r.pose_landmarks[0]]
        for s, e in LINES:
            cv2.line(f, pts[s], pts[e], (0, 255, 0), 3)
        for i in range(11, 33):
            cv2.circle(f, pts[i], 6, (0, 255, 255) if i in LEFT else (255, 0, 255), -1)
    else:
        miss += 1
    cv2.putText(f, f"frame={n} mediapipe", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    wr.write(f)
    if n % 300 == 0: print(f"처리 {n} Frame")
    ok, f = cap.read()
cap.release(); wr.release()
print(f"미검출 {miss}/{n}")
print(f"Saved: {out}")
