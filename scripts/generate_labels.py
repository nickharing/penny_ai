import json
from pathlib import Path

INPUT_METADATA = Path(r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\metadata\metadata.json")
LABEL_DIR = INPUT_METADATA.parent / "labels"

LABEL_FIELDS = ["mint", "reverse_type", "year", "side", "type"]

if __name__ == "__main__":
    with open(INPUT_METADATA, "r") as f:
        entries = json.load(f)

    LABEL_DIR.mkdir(exist_ok=True)

    for field in LABEL_FIELDS:
        values = sorted(set(e[field] for e in entries if field in e))
        label_map = {v: i for i, v in enumerate(values)}

        with open(LABEL_DIR / f"{field}_labels.json", "w") as out:
            json.dump(label_map, out, indent=2)

        print(f"Saved {field} label map with {len(label_map)} entries.")
