# split_dataset.py
import random, shutil
from pathlib import Path

SRC_ROOT = Path("data_raw")
DST_ROOT = Path("data")
DST_ROOT.mkdir(exist_ok=True)

# find top-level folder that contains class dirs, else use SRC_ROOT itself
candidates = [p for p in SRC_ROOT.iterdir() if p.is_dir()]
# If there's exactly one folder and it contains more dirs, use it
if len(candidates) == 1 and any(x.is_dir() for x in candidates[0].iterdir()):
    source = candidates[0]
else:
    source = SRC_ROOT

print("Using source folder:", source)

for cls_dir in source.iterdir():
    if not cls_dir.is_dir():
        continue
    images = [p for p in cls_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    if not images:
        continue
    random.shuffle(images)
    n = len(images)
    n_train = int(0.8 * n) or 1
    n_val = int(0.1 * n)
    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]
    for subset, items in (("train", train), ("val", val), ("test", test)):
        out = DST_ROOT / subset / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy(img, out / img.name)

print("Split finished. See ./data/train ./data/val ./data/test")
