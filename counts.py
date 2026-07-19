# counts.py
from pathlib import Path
root = Path("data")
for split in ("train","val","test"):
    print(f"== {split} ==")
    p = root / split
    if not p.exists():
        print("  missing")
        continue
    for cls in sorted([d.name for d in p.iterdir() if d.is_dir()]):
        cnt = len(list((p/cls).glob("*.*")))
        print(f"  {cls}: {cnt}")
