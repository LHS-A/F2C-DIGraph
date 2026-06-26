"""Check runtime dependencies before training."""

import importlib.util
import sys


REQUIRED = ["torch", "torchvision", "pandas", "numpy", "PIL", "yaml", "tqdm"]
OPTIONAL = ["sklearn", "cv2", "skimage", "transformers"]


def main():
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    optional_missing = [name for name in OPTIONAL if importlib.util.find_spec(name) is None]
    if optional_missing:
        print("Optional packages missing:", ", ".join(optional_missing))
    if missing:
        print("Required packages missing:", ", ".join(missing))
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print("Required dependencies are available.")


if __name__ == "__main__":
    main()
