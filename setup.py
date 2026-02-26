"""
setup.py
--------
Cross-platform setup script for the capstone project.
Creates a virtual environment and installs all Python dependencies.
Works on Windows and macOS (and Linux).

Usage:
    python setup.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def ok(msg):  print(f"{GREEN}  [OK]{RESET} {msg}")
def info(msg): print(f"{YELLOW}  [!!]{RESET} {msg}")
def err(msg):  print(f"{RED}  [ERROR]{RESET} {msg}")


# ── Detect OS ─────────────────────────────────────────────────────────────────

SYSTEM = platform.system()  # "Windows" | "Darwin" | "Linux"


# ── Virtual environment helpers ───────────────────────────────────────────────

def venv_python() -> Path:
    """Return the path to the Python executable inside the venv."""
    if SYSTEM == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def venv_pip() -> Path:
    """Return the path to pip inside the venv."""
    if SYSTEM == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def create_venv():
    if VENV_DIR.exists():
        ok(f"Virtual environment already exists at {VENV_DIR}")
        return
    print(f"\n[1/3] Creating virtual environment at {VENV_DIR}...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    ok("Virtual environment created.")


def install_dependencies():
    print(f"\n[2/3] Installing Python dependencies from requirements.txt...")
    subprocess.check_call([str(venv_pip()), "install", "--upgrade", "pip", "--quiet"])
    subprocess.check_call([str(venv_pip()), "install", "-r", str(REQUIREMENTS)])
    ok("All Python packages installed.")


# ── External tool checks (Tesseract + Poppler) ────────────────────────────────

def check_tesseract():
    """
    Check if Tesseract OCR is installed.
    Only needed for scanned PDFs — digital PDFs work without it.
    """
    print("\n[3/3] Checking external tools...")

    # Try running tesseract --version
    tesseract_cmd = "tesseract"
    if SYSTEM == "Windows":
        win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(win_path):
            tesseract_cmd = win_path

    try:
        result = subprocess.run(
            [tesseract_cmd, "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            version = result.stdout.splitlines()[0]
            ok(f"Tesseract found: {version}")
            return
    except FileNotFoundError:
        pass

    info("Tesseract NOT found. Only needed for scanned PDFs.")
    if SYSTEM == "Darwin":
        info("  Install with: brew install tesseract")
    elif SYSTEM == "Windows":
        info("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        info("  After install, add 'C:\\Program Files\\Tesseract-OCR' to your system PATH.")


def check_poppler():
    """
    Check if Poppler is installed (needed by pdf2image for scanned PDF conversion).
    """
    try:
        result = subprocess.run(
            ["pdftoppm", "-v"],
            capture_output=True, text=True
        )
        if result.returncode == 0 or "Poppler" in (result.stderr + result.stdout):
            ok("Poppler found.")
            return
    except FileNotFoundError:
        pass

    # Windows: check known install locations
    if SYSTEM == "Windows":
        candidates = [
            r"C:\poppler\Library\bin\pdftoppm.exe",
            r"C:\Program Files\poppler\Library\bin\pdftoppm.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                ok(f"Poppler found at: {os.path.dirname(c)}")
                return

    info("Poppler NOT found. Only needed for scanned PDFs.")
    if SYSTEM == "Darwin":
        info("  Install with: brew install poppler")
    elif SYSTEM == "Windows":
        info("  Download from: https://github.com/oschwartz10612/poppler-windows/releases")
        info("  Extract and add the 'bin' folder to your system PATH,")
        info("  OR place it at C:\\poppler\\Library\\bin")


# ── Activation instructions ────────────────────────────────────────────────────

def print_activation_instructions():
    print(f"\n{'='*60}")
    ok("Setup complete!")
    print(f"\n  Activate the virtual environment with:")
    if SYSTEM == "Windows":
        print(f"    venv\\Scripts\\activate")
    else:
        print(f"    source venv/bin/activate")
    print(f"\n  Then run the checker:")
    if SYSTEM == "Windows":
        print(f"    python check.py --instructions directives\\\"Instructions (Directive).docx\" --folder submissions\\")
    else:
        print(f"    python check.py --instructions 'directives/Instructions (Directive).docx' --folder submissions/")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nCapstone Project Setup")
    print(f"  OS      : {SYSTEM} ({platform.machine()})")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Root    : {ROOT}")

    try:
        create_venv()
        install_dependencies()
        check_tesseract()
        check_poppler()
        print_activation_instructions()
    except subprocess.CalledProcessError as e:
        err(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
