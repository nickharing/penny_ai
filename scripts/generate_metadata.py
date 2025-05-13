import os
import re
import json
from glob import glob

DATA_DIR = r"C:\\Users\\nickh\\OneDrive\\Documents\\nh\\Git_repo\\coin_sorter3\\data"
OUT_PATH = r"C:\\Users\\nickh\\OneDrive\\Documents\\nh\\Git_repo\\coin_sorter3\\metadata\\metadata.json"

REVERSE_MAP = {
    'b': 'bicentennial',
    'm': 'memorial',
    'w': 'wheat',
    's': 'shield'
}

def parse_filename(file_path):
    fname = os.path.basename(file_path)
    name, ext = os.path.splitext(fname)
    parts = name.split('_')

    meta = {"filename": fname}

    if parts[0] == "date":
        meta.update({
            "type": "date",
            "year": int(parts[1]),
            "side": parts[2],
            "mint": parts[3],
            "uid": parts[4]
        })
    elif parts[0] in {"mint", "liberty"}:
        meta.update({
            "type": parts[0],
            "year": int(parts[1]),
            "side": parts[2],
            "mint": parts[3],
            "uid": parts[4]
        })
    elif parts[0] == "penny" and parts[1] == "reverse":
        reverse_code = parts[2]
        meta.update({
            "type": "reverse",
            "reverse_type": REVERSE_MAP.get(reverse_code, reverse_code),
            "uid": parts[3]
        })
    elif parts[0] == "penny":
        meta.update({
            "type": "obverse",
            "year": int(parts[1]),
            "side": parts[2],
            "mint": parts[3],
            "uid": parts[4]
        })
    else:
        return None

    return meta


if __name__ == "__main__":
    folders = ["liberty", "mint_mark", "obverse", "reverse", "date"]
    entries = []

    for folder in folders:
        fpath = os.path.join(DATA_DIR, folder)
        for img_path in glob(os.path.join(fpath, "*.jpg")) + glob(os.path.join(fpath, "*.png")):
            meta = parse_filename(img_path)
            if meta:
                entries.append(meta)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Saved {len(entries)} entries to {OUT_PATH}")
