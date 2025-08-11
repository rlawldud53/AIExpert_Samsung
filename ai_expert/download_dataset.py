import os, json, re, shutil, zipfile
from pathlib import Path
from subprocess import run, CalledProcessError
import re
import gdown
import json

def gdown_from_view_url(url: str, out: Path, show_progress: bool = True):
    m = re.search(r"/d/([a-zA-Z0-9_-]+)/", url)
    fid = m.group(1) if m else None
    if not fid:
        raise ValueError("Invalid Drive link: " + url)

    gdown.download(id=fid, output=str(out), quiet=not show_progress)

if __name__ == "__main__":
    CAPTION_FILE_URL = "https://drive.google.com/file/d/1tKgZfjpr9BZWpIS6ikGbSSACdt9NTlkg/view"
    IMAGES_ZIP_URL   = "https://drive.google.com/file/d/1A2dNWabg6_um-V3lhw1tyead5hCpjaW8/view"

    ROOT       = Path("./data/celeba_dialog")
    RAW        = ROOT / "_raw"
    IMAGES_DIR = ROOT / "images"
    TEXT_DIR   = ROOT / "text"
    for p in (ROOT, RAW, IMAGES_DIR, TEXT_DIR):
        p.mkdir(parents=True, exist_ok=True)

    cap_json_path = RAW / "captions.json"
    img_zip_path  = RAW / "images.zip"

    if not img_zip_path.exists():
        gdown_from_view_url(IMAGES_ZIP_URL, img_zip_path)
        print("[OK] images.zip downloaded")
    else:
        print("[SKIP] images.zip exists")

    if not cap_json_path.exists():
        gdown_from_view_url(CAPTION_FILE_URL, cap_json_path)
        print("[OK] captions.json downloaded")
    else:
        print("[SKIP] captions.json exists")
    

    tmp_extract = RAW / "_unzipped"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)

    with zipfile.ZipFile(img_zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for m in tqdm(members, desc="Unzipping", unit="file"):
            z.extract(m, tmp_extract)

    files = [p for ext in ("*.jpg","*.jpeg","*.png") for p in tmp_extract.rglob(ext)]

    moved = 0
    for src in tqdm(files, desc="Organizing images", unit="img"):
        dst = IMAGES_DIR / src.name
        if not dst.exists():             
            shutil.copy2(src, dst)
            moved += 1

    print(f"[OK] images organized: {moved} files")

    jsonl_path = ROOT / "celeba_dialog.jsonl"
    
    if not jsonl_path.exists():
        with open(cap_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imgset = {p.name for ext in ("*.jpg","*.jpeg","*.png") for p in IMAGES_DIR.glob(ext)}
        final = []
        for img, value_dict in data.items():
            if not img or img not in imgset: continue

            if isinstance(value_dict.get("overall_caption"), str):
                caps = [value_dict["overall_caption"].strip()]
            else:
                caps = ["a portrait photo"]
            for c in caps:
                final.append({"image": img, "caption": c})

        if not final:
            final = [{"image": img, "caption": "a portrait photo"} for img in sorted(imgset)]

        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in final))
        print(f"[OK] JSONL written: {len(final)} → {jsonl_path}")

    else:
        print(f"[SKIP] JSONL exists: {jsonl_path}")
