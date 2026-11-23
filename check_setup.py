"""Run this first:  python check_setup.py

It checks everything the experiments need and tells you what to fix. Safe to run many times.
"""
import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).parent
problems = []


def section(title):
    print(f"\n=== {title} ===")


def ok(msg):
    print(f"  [OK]   {msg}")


def bad(msg, fix):
    print(f"  [FAIL] {msg}\n         -> {fix}")
    problems.append(msg)


section("Python")
v = sys.version_info
print(f"  Python {v.major}.{v.minor}.{v.micro} on {platform.system()} {platform.machine()}")
if (3, 9) <= (v.major, v.minor) <= (3, 13):
    ok("Python version is fine")
else:
    bad("Python version is outside 3.9-3.13", "install Python 3.11 or 3.12 from python.org")

section("Packages")
for name in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "transformers"]:
    try:
        mod = importlib.import_module(name)
        ok(f"{name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad(f"{name} is not importable ({type(e).__name__})", f"pip install {'scikit-learn' if name == 'sklearn' else name}")

section("GPU (PyTorch)")
try:
    import torch
    print(f"  torch {torch.__version__} | built for CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        ok(f"GPU found: {props.name}, {props.total_memory / 1e9:.1f} GB memory")
        if props.total_memory / 1e9 < 3.5:
            bad("GPU memory is small (< 3.5 GB)", "DistilBERT may run out of memory; use batch size 16 or Colab")
        x = torch.randn(1000, 1000, device="cuda")
        _ = (x @ x).sum().item()
        ok("a test calculation ran on the GPU")
    else:
        bad("torch cannot see a GPU (cuda.is_available() is False)",
            "install the CUDA build of PyTorch from pytorch.org/get-started/locally, "
            "and update your NVIDIA driver; training on CPU would take far too long")
except Exception as e:
    bad(f"torch is not working ({type(e).__name__}: {str(e)[:80]})", "install PyTorch from pytorch.org/get-started/locally")

section("Data files")
try:
    import pandas as pd
    tr = pd.read_csv(ROOT / "data" / "banking77_train.csv")
    te = pd.read_csv(ROOT / "data" / "banking77_test.csv")
    clinc = json.load(open(ROOT / "data" / "clinc150_full.json"))
    expected = (len(tr), len(te), len(clinc["train"]), len(clinc["val"]), len(clinc["test"]))
    if expected == (10003, 3080, 15000, 3000, 4500):
        ok(f"Banking77 and CLINC150 have the expected sizes {expected}")
    else:
        bad(f"unexpected dataset sizes {expected}", "re-extract the zip; expected (10003, 3080, 15000, 3000, 4500)")
except Exception as e:
    bad(f"cannot read the data folder ({type(e).__name__})", "run this script from inside the chatbot-mldl folder")

section("GloVe (needed for BiLSTM / TextCNN)")
glove = ROOT / "glove" / "glove.6B.300d.txt"
if glove.exists() and glove.stat().st_size > 9e8:
    ok(f"found {glove.name} ({glove.stat().st_size / 1e9:.2f} GB)")
else:
    bad("glove/glove.6B.300d.txt is missing or incomplete", "download glove.6B.zip and extract the 300d file into ./glove/")

section("Hugging Face access (needed once for DistilBERT)")
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ok("downloaded/loaded the DistilBERT tokenizer")
except Exception as e:
    bad(f"could not load DistilBERT files ({type(e).__name__})", "check your internet connection and that transformers is installed")

print("\n" + "=" * 40)
if problems:
    print(f"{len(problems)} problem(s) to fix; see the [FAIL] lines above.")
else:
    print("All checks passed. You are ready to run the experiments.")
