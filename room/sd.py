"""Drop-in for the two `sounddevice` calls run_kermit.py makes, backed by the live room (server.py).

  KERMIT_AUDIO=roomsim  →  run_kermit does `sys.path.insert(0, kermit/room); import sd`
  KERMIT_ROOM_SOCK      →  host:port (default 127.0.0.1:18766) or a unix socket path of the running room

InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=512, callback=cb)  (context manager)
    cb(indata[512, 1], frames, time_info, status) is called from a reader thread as the room's mic
    produces audio (the room is the wall-clock master; 20 ms engine blocks re-cut to `blocksize`).
    time_info.inputBufferAdcTime = perf_counter time of the block's last sample.
OutputStream(samplerate=24000, channels=1, dtype="float32", blocksize=480)  .start() .write() .stop() .close()
    write() blocks like a device: it returns once no more than `latency` seconds are queued ahead of the
    playhead. Each block is sent with its scheduled play time; the room adds its own output latency and
    jitter and plays it through Kermit's loudspeaker into the room (→ echo at the mic).
Only numpy + stdlib: this module is imported inside run_kermit's own venv.
"""
import json, os, socket, struct, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_HDR = struct.Struct("<dI")


def _sock_path():
    return os.environ.get("KERMIT_ROOM_SOCK", "127.0.0.1:18766")


def _connect(role):
    p = _sock_path()
    for _ in range(600):
        try:
            if ":" in p:
                h, port = p.rsplit(":", 1)
                s = socket.create_connection((h, int(port))); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            else:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(p)
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError(f"room not running at {_sock_path()}")
    s.sendall((json.dumps({"role": role}) + "\n").encode())
    return s


def _recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c: raise EOFError
        b += c
    return bytes(b)


class _T:
    def __init__(self, t): self.inputBufferAdcTime = t; self.currentTime = time.perf_counter()


class InputStream:
    def __init__(self, samplerate=16000, blocksize=512, channels=1, dtype="float32", callback=None, **kw):
        assert samplerate == 16000 and channels == 1, "room mic is 16 kHz mono"
        self.bs, self.cb = blocksize, callback
        self._run = False

    def start(self):
        self._s = _connect("mic"); self._run = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf = np.zeros(0, np.float32)
        try:
            while self._run:
                t_end, n = _HDR.unpack(_recv_exact(self._s, _HDR.size))
                x = np.frombuffer(_recv_exact(self._s, 4 * n), np.float32)
                buf = np.concatenate([buf, x])
                while len(buf) >= self.bs:
                    blk, buf = buf[:self.bs], buf[self.bs:]
                    self.cb(blk.reshape(-1, 1).copy(), self.bs, _T(t_end - len(buf) / 16000), None)
        except (EOFError, OSError):
            pass

    def stop(self): self._run = False
    def close(self):
        self._run = False
        try: self._s.close()
        except Exception: pass

    def __enter__(self): self.start(); return self
    def __exit__(self, *e): self.close()


class OutputStream:
    def __init__(self, samplerate=24000, channels=1, dtype="float32", blocksize=480, latency=0.04, **kw):
        assert samplerate == 24000 and channels == 1, "room loudspeaker is fed 24 kHz mono"
        self.sr, self.latency = samplerate, latency
        self.t_next = None
        self._s = None

    def start(self):
        self._s = _connect("spk")

    def write(self, data):
        x = np.ascontiguousarray(np.asarray(data, np.float32).reshape(-1))
        now = time.perf_counter()
        if self.t_next is None or self.t_next < now:      # (re)start after an underrun: device buffer refills
            self.t_next = now + self.latency
        self._s.sendall(_HDR.pack(self.t_next, len(x)) + x.tobytes())
        self.t_next += len(x) / self.sr
        wait = self.t_next - self.latency - time.perf_counter()  # keep ~latency of audio queued, like a device
        if wait > 0: time.sleep(wait)

    def stop(self): pass
    def close(self):
        try: self._s.close()
        except Exception: pass

    def __enter__(self): self.start(); return self
    def __exit__(self, *e): self.close()
