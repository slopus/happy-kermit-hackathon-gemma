"""`import sounddevice` → the live room (kermit/room/sd.py). Put kermit/room/shim first on PYTHONPATH to run
run_kermit.py unchanged inside the simulated room (until TURN's KERMIT_AUDIO=roomsim switch lands)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd import InputStream, OutputStream  # noqa: E402,F401
