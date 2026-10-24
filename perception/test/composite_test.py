"""Paste real face photos into the real camera snapshot at known 3D positions; check detection + geometry.
Also: hysteresis (two faces), dropout → present:false timing."""
import json, math, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import track

bg = cv2.imread(os.path.join(HERE, "snap_0.jpg")); H, W = bg.shape[:2]
args = track.parse_args([])
cam = track.Camera(W, H, args.hfov, pos=args.cam_pos, rpy=args.cam_rpy)
det = cv2.FaceDetectorYN.create(track.MODEL, "", (0, 0), 0.7, 0.3, 50)

def face_box(img):
    det.setInputSize((img.shape[1], img.shape[0])); _, d = det.detect(img); return d[0][:4]

srcs = {n: cv2.imread(os.path.join(HERE, n)) for n in ["lena.jpg", "messi5.jpg"]}
boxes = {n: face_box(im) for n, im in srcs.items()}

def paste(canvas, name, u, v, w_px):
    """Scale source so its detected face is w_px wide and centred at (u, v); paste a soft ellipse around the head."""
    im, (x, y, bw, bh) = srcs[name], boxes[name]
    s = w_px / bw; im2 = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    fx, fy = (x + bw / 2) * s, (y + bh / 2) * s
    ox, oy = int(round(u - fx)), int(round(v - fy))
    mask = np.zeros(im2.shape[:2], np.float32)
    cv2.ellipse(mask, (int(fx), int(fy)), (int(bw * s * 1.0), int(bh * s * 1.3)), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1, bw * s * 0.08))[..., None]
    x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + im2.shape[1], W), min(oy + im2.shape[0], H)
    if x1 <= x0 or y1 <= y0: return canvas
    sub = im2[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; m = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1 - m) + sub * m).astype(np.uint8)
    return canvas

def project(p_base):
    """Base-frame point → pixel (u, v) and face width px (inverse of Camera.face_to_base)."""
    pb = cam.R.T @ (np.array(p_base) - cam.pos); z = pb[0]
    return cam.cx - pb[1] / z * cam.fx, cam.cy - pb[2] / z * cam.fy, cam.fx * cam.face_w / z

# 1. accuracy sweep
rows = []; os.makedirs(os.path.join(HERE, "composite"), exist_ok=True)
for name in srcs:
    for d in [0.7, 1.0, 1.5, 2.0, 3.0]:
        for yaw in [-35, -20, 0, 20, 35]:
            pitch = 8.0
            p = np.array([d * math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)),
                          d * math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)), d * math.sin(math.radians(pitch))]) + 0
            u, v, wpx = project(p)
            if not (wpx / 2 < u < W - wpx / 2 and wpx / 2 < v < H - wpx / 2): continue
            img = paste(bg.copy(), name, u, v, wpx)
            tr = track.Tracker(args); m = tr.step(img, t=0.0)
            if (d, yaw) in [(1.5, -35), (1.0, 20)] and name == "lena.jpg": cv2.imwrite(os.path.join(HERE, "composite", f"lena_d{d}_y{yaw}.jpg"), img)
            truth_yaw, truth_pitch = track.bearing(p)
            rows.append({"src": name, "d": d, "yaw": yaw, "wpx": round(wpx, 1), "found": m["present"],
                         "yaw_err": round(m["yaw_deg"] - truth_yaw, 2) if m["present"] else None,
                         "pitch_err": round(m["pitch_deg"] - truth_pitch, 2) if m["present"] else None,
                         "dist_err_pct": round(100 * (m["dist_m"] / np.linalg.norm(p) - 1), 1) if m["present"] else None})
found = [r for r in rows if r["found"]]
print(f"detected {len(found)}/{len(rows)}; misses: {[(r['src'][:4], r['d'], r['yaw'], r['wpx']) for r in rows if not r['found']]}")
for k in ["yaw_err", "pitch_err", "dist_err_pct"]:
    v = np.abs([r[k] for r in found]); print(f"  |{k}| median {np.median(v):.2f}  max {v.max():.2f}")
json.dump(rows, open(os.path.join(HERE, "composite_results.json"), "w"), indent=0)

# 2. hysteresis + dropout: A (lena) at 1.5 m left; B (messi) approaches from 3 m to 1.0 m on the right; then both vanish
tr = track.Tracker(args); log = []
for i in range(int(6 / (1 / 15))):
    t = i / 15; img = bg.copy()
    if t < 4.5:
        uA, vA, wA = project([1.5, 0.5, 0.15]); img = paste(img, "lena.jpg", uA, vA, wA)
        dB = max(1.0, 3.0 - t * 0.8); uB, vB, wB = project([dB, -0.6 * dB / 1.5, 0.15]); img = paste(img, "messi5.jpg", uB, vB, wB)
    m = tr.step(img, t=t); log.append((round(t, 2), m.get("present"), m.get("yaw_deg"), m.get("dist_m"), m.get("n_faces"), round(wB / wA, 2) if t < 4.5 else None))
switch = next((l for l in log if l[2] is not None and l[2] < 0), None)
first_ratio = next((l for l in log if l[5] and l[5] > 1.3), None)
gone = next((l for l in log if l[0] >= 4.5 and l[1] is False), None)
print("B >1.3x larger from t =", first_ratio[0], "| tracker switched to B at t =", switch[0], "| faces gone t=4.5 → present:false at t =", gone[0])
for l in log[::6]: print("  ", l)
