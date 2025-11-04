"""
Simple rule-based diagram parser baseline (robust)
- Fixes 'single positional indexer is out-of-bounds'
- Hardens OCR filtering and arrow handling
"""

import os
import cv2, pytesseract, pandas as pd, numpy as np

# === CONFIG ===
IMAGE_DIR = "../data/CBM_data/images_CBM_subset"
OUTPUT_CSV = "rule_based_triples.csv"
AREA_THRESH = 50           # ignore tiny contours
OCR_CONF_THRESH = 60       # minimum OCR confidence
TESSERACT_CONFIG = "--psm 6"   # good default for diagrams

def nearest_label(point, texts):
    """Return nearest recognized text label to a given (x,y) point."""
    if texts.empty:
        return "N/A"
    dx = texts["cx"] - point[0]
    dy = texts["cy"] - point[1]
    d = (dx*dx + dy*dy)
    # use .idxmin with .loc (label-based), NOT iloc
    idx = d.idxmin()
    return texts.loc[idx, "text"]

def parse_image(img_path):
    """Process one image and return list of (image, subject, relation, object) triples."""
    img = cv2.imread(img_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # light denoise/binarize to help OCR
    gray = cv2.medianBlur(gray, 3)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(bin_img, 100, 200)

    # --- Arrow-ish contour detection (very rough) ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrows = [c for c in contours if cv2.contourArea(c) > AREA_THRESH]

    # --- OCR text detection ---
    ocr_df = pytesseract.image_to_data(
        bin_img,
        output_type=pytesseract.Output.DATAFRAME,
        config=TESSERACT_CONFIG
    )
    if ocr_df is None or ocr_df.empty or "conf" not in ocr_df:
        texts = pd.DataFrame(columns=["text","left","top","width","height","cx","cy"])
    else:
        # make confidence numeric; drop low/invalid rows
        ocr_df["conf"] = pd.to_numeric(ocr_df["conf"], errors="coerce")
        texts = (
            ocr_df[
                (ocr_df["conf"] >= OCR_CONF_THRESH) &
                ocr_df["text"].notna()
            ][["text","left","top","width","height"]]
            .copy()
        )
        # clean text and drop empty strings
        texts["text"] = texts["text"].str.strip()
        texts = texts[texts["text"].str.len() > 0]

        # compute centers; coerce numeric (Tesseract sometimes returns strings)
        for col in ["left","top","width","height"]:
            texts[col] = pd.to_numeric(texts[col], errors="coerce")
        texts = texts.dropna(subset=["left","top","width","height"])
        texts["cx"] = texts["left"] + texts["width"]/2.0
        texts["cy"] = texts["top"] + texts["height"]/2.0

        # CRITICAL: reset index so .idxmin aligns with .loc labels
        texts = texts.reset_index(drop=False)  # keep old index in 'index' if you want
        texts = texts.set_index(texts.index)   # labels now 0..N-1

    triples = []
    if len(arrows) == 0 or texts.empty:
        return triples  # nothing to extract

    for cnt in arrows:
        x, y, w, h = cv2.boundingRect(cnt)

        # crude tail/head heuristic: left→right
        tail = (x, y + h // 2)
        head = (x + w, y + h // 2)

        s = nearest_label(tail, texts)
        o = nearest_label(head, texts)

        # skip if we failed to get two labels
        if s == "N/A" or o == "N/A":
            continue

        # normalize minus variants
        def has_minus(t): 
            return any(ch in t for ch in ["−", "–", "-"])

        predicate = (
            "increases" if ("+" in s or "+" in o)
            else "decreases" if (has_minus(s) or has_minus(o))
            else "interacts_with"
        )
        triples.append((os.path.basename(img_path), s, predicate, o))

    return triples

# === Main loop ===
all_triples = []
images = [f for f in os.listdir(IMAGE_DIR)
          if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff",".bmp"))]

print(f"Found {len(images)} images in {IMAGE_DIR}")
for fname in images:
    fpath = os.path.join(IMAGE_DIR, fname)
    try:
        t = parse_image(fpath)
        all_triples.extend(t)
        print(f"[✓] Parsed {fname}: {len(t)} triples")
    except Exception as e:
        print(f"[x] Error on {fname}: {e}")

# === Save results ===
if all_triples:
    df = pd.DataFrame(all_triples, columns=["image","subject","relation","object"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} triples to {OUTPUT_CSV}")
else:
    print("No triples extracted.")
