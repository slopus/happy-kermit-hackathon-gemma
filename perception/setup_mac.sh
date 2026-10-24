#!/bin/bash
# PERCEPTION v2 one-time setup on the Mac (after voice/setup_mac.sh). Needs network ONCE; everything after runs offline.
#   cd kermit/perception && ./setup_mac.sh
# Installs into the voice venv (run_kermit.py runs there) and fetches ~95 MB of models into perception/models.
set -euo pipefail
cd "$(dirname "$0")"
VENV=../voice/.venv
[ -x $VENV/bin/python ] || { echo "run ../voice/setup_mac.sh first"; exit 1; }
VIRTUAL_ENV=$VENV uv pip install "ultralytics>=8.3" lap python_speech_features scipy "torch>=2.5" torchvision

mkdir -p models
R=https://github.com/ultralytics/assets/releases/download/v8.3.0
[ -f models/yolo11n-pose.pt ] || curl -L -o models/yolo11n-pose.pt $R/yolo11n-pose.pt                # 6 MB  people + 17 keypoints
[ -f models/face_detection_yunet_2023mar.onnx ] || curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx   # 230 kB
[ -f models/pretrain_AVA_CVPR.model ] || curl -L -o models/pretrain_AVA_CVPR.model \
  https://github.com/Junhua-Liao/Light-ASD/raw/main/weight/pretrain_AVA_CVPR.model                    # 4 MB  Light-ASD
# Mic detector: YOLO-World-m with the "microphone" prompts baked in, so no CLIP is needed at runtime.
# (If you copied models/ from the Linux box, yolov8m-worldv2-mic.pt is already there and this is skipped.)
if [ ! -f models/yolov8m-worldv2-mic.pt ]; then
  curl -L -o models/yolov8m-worldv2.pt $R/yolov8m-worldv2.pt                                        # 57 MB
  YOLO_CONFIG_DIR=.ultralytics $VENV/bin/python - <<'EOF'
from ultralytics import YOLOWorld                      # downloads CLIP ViT-B/32 once (~340 MB, cached) to embed the prompts
m = YOLOWorld("models/yolov8m-worldv2.pt"); m.set_classes(["microphone", "handheld microphone"]); m.save("models/yolov8m-worldv2-mic.pt")
EOF
fi
# warm-up + MPS check (also triggers ultralytics' one-time checks while online)
YOLO_CONFIG_DIR=.ultralytics $VENV/bin/python - <<'EOF'
import numpy as np, sys; sys.path.insert(0, ".")
import people
pp = people.People(people.parse_args(["--device", "mps"]))
for i in range(5): pp.step(np.zeros((720, 1280, 3), np.uint8), i / 30)
print("perception OK on", pp.device, {k: round(float(np.median(v)), 1) for k, v in pp.timing.items()})
EOF
echo "perception setup done. smoke test: $VENV/bin/python people.py --source 0 --hfov 78 --stats"
