"""Dump raw YOLO-World mic detections + pose on sampled frames for the mic eval (eval/mic/*.json)."""
import os, sys, json, subprocess
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), "..", ".ultralytics"))
import numpy as np, cv2
from ultralytics import YOLO, YOLOWorld
M = lambda f: os.path.join(os.path.dirname(__file__), "..", "models", f)
pose = YOLO(M("yolo11n-pose.pt")); mics = {k: YOLOWorld(M(f"yolov8{k}-worldv2-mic.pt")) for k in ("s", "m")}
out = []
for spec in sys.argv[1:]:
    path, step, n = spec.split(":"); step, n = float(step), int(n)
    cap = cv2.VideoCapture(path)
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, i * step * 1000); ok, img = cap.read()
        if not ok: break
        r = pose.predict(img, conf=0.35, verbose=False)[0]
        ppl = [{"box": b.tolist(), "kp": k.tolist()} for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy())]
        rec = {"src": os.path.basename(path), "i": i, "t": i * step, "people": ppl, "w": img.shape[1], "h": img.shape[0]}
        for k, m in mics.items():
            d = m.predict(img, conf=0.05, imgsz=640, verbose=False)[0]
            rec["mic_" + k] = [{"box": b.xyxy[0].tolist(), "conf": float(b.conf), "cls": int(b.cls)} for b in d.boxes]
        out.append(rec)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "mic", "dump.json"), "w"))
print(len(out))
