"""Light-ASD (Liao et al., CVPR 2023, MIT licence; vendored encoder/GRU code), ~1.0 M params.

Input: grayscale 112×112 face crops at 25 fps + 13-dim MFCC at 100 fps (4 audio frames per video frame).
Output: one logit per video frame (>0 = speaking). Weights: models/pretrain_AVA_CVPR.model (AVA-trained).
"""
import os
import numpy as np, torch, torch.nn as nn
from .Encoder import visual_encoder, audio_encoder
from .Classifier import BGRU

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "models", "pretrain_AVA_CVPR.model")
CROP_MODE = os.environ.get("KERMIT_ASD_CROP", "pad")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualEncoder, self.audioEncoder, self.GRU = visual_encoder(), audio_encoder(), BGRU(128)
        self.FC = nn.Linear(128, 2)

    def forward(self, a, v):                      # a: B×4T×13, v: B×T×112×112 (0..255)
        B, T = v.shape[:2]
        ea = self.audioEncoder(a.unsqueeze(1).transpose(2, 3))
        ev = self.visualEncoder(((v / 255 - 0.4161) / 0.1688).view(B, 1, T, 112, 112))
        x = self.GRU(ea + ev).reshape(-1, 128)
        return self.FC(x)[:, 1].view(B, T)       # same as Light-ASD's lossAV inference score


def pick_device(pref=None):
    if pref: return pref
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


class LightASD:
    def __init__(self, device=None, weights=WEIGHTS):
        self.device = pick_device(device)
        self.m = _Model()
        sd = torch.load(weights, map_location="cpu")
        own = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
        own.update({k.replace("lossAV.", ""): v for k, v in sd.items() if k.startswith("lossAV.FC")})
        self.m.load_state_dict(own)
        self.m.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, faces, mfcc):
        """faces: B×T×112×112 uint8; mfcc: 4T×13 (shared audio) or B×4T×13. Returns B×T float logits."""
        v = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(mfcc, np.float32), device=self.device)
        if a.dim() == 2: a = a.unsqueeze(0).expand(v.shape[0], -1, -1)
        return self.m(a, v).float().cpu().numpy()


def mfcc(audio16, sr=16000):
    """Same features Light-ASD/TalkNet use (python_speech_features defaults, 100 fps). audio: int16-scale or float."""
    import python_speech_features
    x = np.asarray(audio16, np.float32)
    if x.dtype != np.int16 and np.abs(x).max(initial=0) <= 1.5: x = x * 32768.0
    return python_speech_features.mfcc(x, sr, numcep=13, winlen=0.025, winstep=0.010)


def crop_face(gray, cx, cy, size, cs=0.40, mode=None):
    """mode "pad" (Light-ASD/TalkNet demo + Columbia eval): square box half-size s=max(w,h)/2, padded by cs, resized
    to 224, centre 112×112 (mouth-centred). mode "box" (AVA training): the square face box itself resized to 112.
    gray: H×W uint8 image. size = max(face w, face h) in px."""
    mode = mode or CROP_MODE
    if mode == "box":
        import cv2
        h, w = gray.shape; s = size / 2
        y0, y1, x0, x1 = int(cy - s), int(cy + s), int(cx - s), int(cx + s)
        pad = np.full((max(1, y1 - y0), max(1, x1 - x0)), 110, np.uint8)
        sy0, sy1, sx0, sx1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
        if sy1 > sy0 and sx1 > sx0: pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
        return cv2.resize(pad, (112, 112))
    bs = size / 2
    h, w = gray.shape
    y0, y1 = int(cy - bs), int(cy + bs * (1 + 2 * cs)); x0, x1 = int(cx - bs * (1 + cs)), int(cx + bs * (1 + cs))
    pad = np.full((y1 - y0, x1 - x0), 110, np.uint8)
    sy0, sy1, sx0, sx1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    if sy1 > sy0 and sx1 > sx0: pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    import cv2
    f = cv2.resize(pad, (224, 224))
    return f[56:168, 56:168]
