"""MeetAssist.exe - tiny launcher. Starts the bundled portable Python (runtime\python\pythonw.exe) running
meetassist.app, with no console window. All real code lives in ./meetassist and ./runtime next to this exe."""
import ctypes
import os
import subprocess
import sys

root = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
if not getattr(sys, "frozen", False):
    root = os.path.dirname(root)
py = os.path.join(root, "runtime", "python", "pythonw.exe")


def fail(msg: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, msg, "MeetAssist", 0x10)
    sys.exit(1)


if not os.path.exists(py):
    fail("MeetAssist cannot start: the 'runtime' folder is missing next to MeetAssist.exe.\n\n"
         "Copy the whole MeetAssist folder, not only the .exe.")
if not os.path.isdir(os.path.join(root, "meetassist")):
    fail("MeetAssist cannot start: the 'meetassist' folder is missing next to MeetAssist.exe.")

env = dict(os.environ)
env["PYTHONPATH"] = root
env["PYTHONNOUSERSITE"] = "1"          # never pick up packages from the user's own Python installs
env["PYTHONDONTWRITEBYTECODE"] = "1"
subprocess.Popen([py, "-m", "meetassist.app"], cwd=root, env=env, creationflags=0x08000000 | 0x00000008)
