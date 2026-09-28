"""
setup_check.py
--------------
Validates the project environment by:
  1. Importing and printing versions of all required packages
  2. Checking CUDA / GPU availability
  3. Running a small LSTM smoke-test with a random tensor
"""

import sys
import os

# Force UTF-8 output so emoji render correctly on all terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# 1.  Package version checks
# ─────────────────────────────────────────────

REQUIRED = [
    ("torch",      "torch"),
    ("pandas",     "pandas"),
    ("numpy",      "numpy"),
    ("sklearn",    "scikit-learn"),
    ("streamlit",  "streamlit"),
    ("plotly",     "plotly"),
    ("shap",       "shap"),
]

failed_imports = []

print("=" * 55)
print("  [PKG]  PACKAGE VERSION REPORT")
print("=" * 55)

modules = {}
for import_name, pip_name in REQUIRED:
    try:
        mod = __import__(import_name)
        if import_name == "sklearn":
            import sklearn
            version = sklearn.__version__
        else:
            version = getattr(mod, "__version__", "unknown")
        modules[import_name] = mod
        print(f"  [OK]  {pip_name:<18}  v{version}")
    except ImportError as e:
        print(f"  [!!]  {pip_name:<18}  MISSING  ({e})")
        failed_imports.append((import_name, pip_name))

print()

# ─────────────────────────────────────────────
# 2.  CUDA / GPU availability
# ─────────────────────────────────────────────

print("=" * 55)
print("  [HW]   HARDWARE CHECK")
print("=" * 55)

if "torch" not in [f[0] for f in failed_imports] and "torch" in modules:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [OK]  GPU available  ->  {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        print("  [--]  CPU only -- training will be slower")
else:
    print("  [!!]  Cannot check GPU -- torch failed to import")

print()

# ─────────────────────────────────────────────
# 3.  PyTorch LSTM smoke-test
# ─────────────────────────────────────────────

print("=" * 55)
print("  [TEST] PYTORCH LSTM SMOKE TEST")
print("=" * 55)

lstm_ok = False
if "torch" not in [f[0] for f in failed_imports]:
    try:
        import torch
        import torch.nn as nn

        # Tensor shape: (batch=4, seq_len=20, input_size=18)
        x = torch.randn(4, 20, 18)
        print(f"  Input tensor   : shape {list(x.shape)}  dtype {x.dtype}")

        lstm = nn.LSTM(
            input_size=18,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        lstm.eval()

        with torch.no_grad():
            output, (h_n, c_n) = lstm(x)

        print(f"  Output tensor  : shape {list(output.shape)}")
        print(f"  Hidden state   : shape {list(h_n.shape)}")
        print(f"  Cell state     : shape {list(c_n.shape)}")
        print("  [OK]  LSTM forward pass succeeded")
        lstm_ok = True
    except Exception as exc:
        print(f"  [!!]  LSTM test FAILED: {exc}")
else:
    print("  [!!]  Skipped -- torch not available")

print()

# ─────────────────────────────────────────────
# 4.  Final verdict
# ─────────────────────────────────────────────

print("=" * 55)
if not failed_imports and lstm_ok:
    print("  \u2705  ALL SYSTEMS GO")
else:
    print("  \u274c  ISSUES DETECTED:")
    for import_name, pip_name in failed_imports:
        print(f"       * {pip_name} could not be imported -- run: pip install {pip_name}")
    if not lstm_ok and not failed_imports:
        print("       * PyTorch LSTM smoke test failed (see details above)")
print("=" * 55)

sys.exit(0 if (not failed_imports and lstm_ok) else 1)
