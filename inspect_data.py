# inspect_data.py
from pathlib import Path
from PIL import Image, ImageOps
import math
import numpy as np

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def get_image_files(folder):
    return [p for p in folder.glob("*.*") if p.is_file()]

def check_folder(folder):
    print(f"\n=== {folder} ===")
    if not folder.exists():
        print("  (missing)")
        return
    classes = sorted([d for d in folder.iterdir() if d.is_dir()])
    if not classes:
        print("  (no class subfolders)")
        return
    for cls in classes:
        files = get_image_files(cls)
        print(f"  {cls.name}: {len(files)} images")
        # check first few images for size/corruption
        problems = []
        small = []
        samples = []
        for i, f in enumerate(files):
            if i >= 40:  # just look at up to 40 files for speed
                break
            try:
                with Image.open(f) as im:
                    im.verify()  # detect truncated/corrupt images
                # reopen to get size (verify() can close file)
                with Image.open(f) as im:
                    w,h = im.size
                    if w < 50 or h < 50:
                        small.append((f, (w,h)))
                    # collect sample thumbnails (first 9)
                    if len(samples) < 9:
                        with Image.open(f) as im2:
                            imthumb = ImageOps.fit(im2.convert("RGB"), (224,224))
                            samples.append(imthumb)
            except Exception as e:
                problems.append((f, str(e)))
        if problems:
            print("    !!! Corrupt images found (first 5):")
            for p in problems[:5]:
                print("      ", p[0].name, "->", p[1])
        if small:
            print("    !!! Very small images found (first 5):")
            for s in small[:5]:
                print("      ", s[0].name, "size=", s[1])
        # save sample grid if we collected any samples
        if samples:
            cols = 3
            rows = math.ceil(len(samples)/cols)
            grid = Image.new("RGB", (cols*224, rows*224))
            for idx, im in enumerate(samples):
                x = (idx % cols) * 224
                y = (idx // cols) * 224
                grid.paste(im, (x,y))
            outp = OUT_DIR / f"samples_grid_{folder.name}_{cls.name}.jpg"
            grid.save(outp)
            print("    Saved sample grid:", outp)

def main():
    print("Inspecting data folders..")
    check_folder(TRAIN_DIR)
    check_folder(VAL_DIR)
    check_folder(TEST_DIR)
    print("\nDone. Look in the 'results' folder for any 'samples_grid_*.jpg' to visually inspect.")

if __name__ == "__main__":
    main()
