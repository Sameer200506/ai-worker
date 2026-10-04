"""Create a local Python virtual environment."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
venv = root / ".venv"
if not venv.exists():
    subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
pip = venv / ("Scripts/pip.exe" if sys.platform.startswith("win") else "bin/pip")
subprocess.check_call([str(pip), "install", "-r", str(root / "backend" / "requirements.txt")])
print("Virtual environment ready:", venv)
