# make_dummy_data.py
from pathlib import Path
from PIL import Image
import numpy as np

root = Path("data")
classes = ["healthy","diseased"]

for split in ["train","val","test"]:
    for cls in classes:
        out = root / split / cls
        out.mkdir(parents=True, exist_ok=True)
        # create few images
        n = 8 if split=="train" else 2
        for i in range(n):
            arr = (np.random.rand(224,224,3) * 255).astype("uint8")
            img = Image.fromarray(arr)
            img.save(out / f"{cls}_{i}.jpg")

print("Dummy dataset created successfully!")
