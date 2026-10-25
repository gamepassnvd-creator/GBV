# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — portable one-dir bundle for the GBV Monitor.

Build:
    python -m pyinstaller packaging/gbv_monitor.spec --noconfirm

Then assemble the distribution folder:
    dist/GBVMonitor/GBVMonitor.exe  + _internal/   <- produced here
      runtime/<backend>/…   <- copy from .bundle/runtime/<backend>/
      models/*.gguf         <- copy from .bundle/models/
      ms-playwright/…       <- copy from .bundle-browser/ (PLAYWRIGHT_BROWSERS_PATH)
      output/               <- created at first run

`packaging/build_windows.ps1` and the GitHub Actions workflow do all of that
for you.
"""
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

APP_DIR = os.path.abspath(os.path.join(SPECPATH, ".."))

# ------------------------------------------------------------------ app sources
datas = [
    ("app.py", "."),
    ("analyzer.py", "."),
    ("alerts.py", "."),
    ("agent.py", "."),
    ("lmstudio.py", "."),
    ("local_llm.py", "."),
    ("launcher.py", "."),
    ("monitor.py", "."),
    ("osint.py", "."),
    ("pdf_report.py", "."),
    ("report.py", "."),
    ("scraper.py", "."),
    ("storage.py", "."),
    ("packaging/DIST_README.txt", ".."),      # lands next to the exe as README
]

binaries = []
hiddenimports = ["uharfbuzz", "fpdf", "networkx", "PIL", "altair", "pyarrow"]

# Big/lazy packages that plain analysis cannot fully map.
for pkg in ("streamlit", "playwright", "altair", "pyarrow", "networkx"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ------------------------------------------------------------------ analysis
a = Analysis(
    ["packaging/app_entry.py"],
    pathex=[APP_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "scipy", "sympy", "sqlalchemy",
        "jupyter", "notebook", "IPython", "pyspark", "torch", "sklearn",
        "PyQt5", "PyQt6", "wx", "pytest",
    ],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GBVMonitor",
    debug=False,
    strip=False,
    upx=False,
    console=True,        # visible console on purpose: model loading progress & errors
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GBVMonitor",
)
