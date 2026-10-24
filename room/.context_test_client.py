import sys, time, threading, numpy as np, soundfile as sf
sys.path.insert(0, '.'); import sd
from scipy.signal import resample_poly
got = []
def cb(x, n, t, s): got.append((t.inputBufferAdcTime, x[:,0].copy()))
x, sr = sf.read('../voice/samples/open_1.wav', dtype='float32')
if sr != 24000: x = resample_poly(x, 24000, sr).astype(np.float32)
with sd.InputStream(samplerate=16000, blocksize=512, channels=1, dtype='float32', callback=cb):
    o = sd.OutputStream(samplerate=24000, channels=1, dtype='float32', blocksize=480); o.start()
    time.sleep(3)
    t_w = time.perf_counter()
    for i in range(0, len(x), 480): o.write(x[i:i+480].reshape(-1,1))
    print('wrote', len(x)/24000, 's in', round(time.perf_counter()-t_w, 3))
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
ts = np.array([t for t,_ in got]); print('mic blocks', len(got), 'dt mean', np.diff(ts).mean(), 'jitter max', np.abs(np.diff(ts)-0.032).max())
