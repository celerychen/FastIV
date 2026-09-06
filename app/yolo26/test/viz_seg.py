#!/usr/bin/env python3
"""YOLO26-seg visualization on bus.jpg.

Conventions (all verified against the reference dumps):
  - detect_out rows are [x1,y1,x2,y2,score,cls, coef(nm=32)], coordinates in
    the 320x320 LETTERBOX grid, ranked by score desc.
  - proto[32,80,80] covers the whole 320 letterbox grid (stride 4).
  - original image pixel (ox,oy) sits inside the letterbox content region at
    L = (dx + ox*r, dy + oy*r); mask grid sample g = L/4.
  - mask_gt{i} = sigmoid(sum_nm coef[nm]*proto[nm]) for detect_out row i
    (rows 0..7), so rows i<8 are self-checked pixel-wise here.
"""
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

R = "app/yolo26/models/yolo26n-seg/"
IMG = "app/yolo26/test/real/bus.jpg"
OUT = "app/yolo26/test/real/bus_seg_C.png"
if len(sys.argv) >= 2:
    R = sys.argv[1] if sys.argv[1].endswith("/") else sys.argv[1] + "/"
if len(sys.argv) >= 3:
    IMG = sys.argv[2]
if len(sys.argv) >= 4:
    OUT = sys.argv[3]
W_IN = H_IN = 320

def ld(n):
    d = json.load(open(R + n + ".json"))
    return np.fromfile(R + n + ".f32", dtype="<f4").reshape(d["shape"])

names = json.load(open(R + "names.json"))
de = ld("detect_out")[0]                 # [300, 38]
proto = ld("proto")[0].astype(np.float64)  # [32, 80, 80]
nm = proto.shape[0]
Hp, Wp = proto.shape[1], proto.shape[2]

img0 = Image.open(IMG).convert("RGB")
iw, ih = img0.size
r = min(W_IN / iw, H_IN / ih)
nw, nh = max(1, round(iw * r)), max(1, round(ih * r))
dx, dy = (W_IN - nw) // 2, (H_IN - nh) // 2
print(f"{os.path.basename(IMG)} {iw}x{ih}  letterbox r={r:.5f} content={nw}x{nh} pad=({dx},{dy})")

def to_orig(b):
    return ((b[0] - dx) / r, (b[1] - dy) / r, (b[2] - dx) / r, (b[3] - dy) / r)

# ---- self-check: rebuild rows 0..7 vs mask_gt (top-k row i <-> mask_gt{i}) ----
def rebuild(row_coef):
    acc = np.zeros((Hp, Wp), dtype=np.float64)
    for m in range(nm):
        acc += row_coef[m] * proto[m]
    return 1.0 / (1.0 + np.exp(-acc))

for i in range(min(8, de.shape[0])):
    gt = ld(f"mask_gt{i}")[0]
    me = rebuild(de[i, 6:6 + nm].astype(np.float64))
    err = float(np.abs(me - gt).max())
    print(f"  selfcheck row{i}: mask rebuild vs mask_gt{i} max_abs_err={err:.2e}")
    assert err < 1e-2, f"row {i} rebuild mismatch"

# ---- detections to draw (score >= 0.25) ----
o = np.argsort(-de[:, 4])
rows = [de[i] for i in o if de[i, 4] >= 0.25]
print(f"detections drawn (score>=0.25): {len(rows)}")

palette = {0: (235, 64, 64), 5: (70, 140, 235), 15: (235, 170, 40), 2: (150, 90, 240)}

# full-resolution mask per original pixel: build via 320-letterbox then content->orig
def mask_to_orig(m80):
    m320 = np.asarray(Image.fromarray((m80 * 255).astype(np.uint8), "L")
                      .resize((W_IN, H_IN), Image.BILINEAR), dtype=np.float32) / 255.0
    content = m320[dy:dy + nh, dx:dx + nw]                    # content region in letterbox
    content = np.asarray(Image.fromarray((content * 255).astype(np.uint8), "L")
                         .resize((iw, ih), Image.BILINEAR), dtype=np.float32) / 255.0
    return content

overlay = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
# draw per instance, small objects (persons) on top of large (bus):
# iterate by DECREASING area so the bus goes first and persons composite last
order = sorted(range(len(rows)), key=lambda k: (rows[k][2] - rows[k][0]) * (rows[k][3] - rows[k][1]), reverse=True)

for k in order:
    bb = rows[k]
    cls = int(round(bb[5]))
    col = palette.get(cls, (130, 190, 130))
    m = rebuild(bb[6:6 + nm].astype(np.float64))
    om = mask_to_orig(m)                                      # [ih, iw] soft mask
    # per-instance colored layer, alpha proportional to mask (om in [0,1] ->
    # scale to 0..255 first; the earlier om*0.55 then uint8 truncation made
    # every alpha 0 and the mask invisible)
    lay = Image.new("RGBA", (iw, ih), col + (255,))
    alpha = np.clip(om * 255.0 * 0.55, 0, 255).astype(np.uint8)
    lay.putalpha(Image.fromarray(alpha, "L"))
    overlay = Image.alpha_composite(overlay, lay)

# boxes + labels
d = ImageDraw.Draw(overlay)
for bb in rows:
    cls = int(round(bb[5]))
    col = palette.get(cls, (130, 190, 130))
    x1, y1, x2, y2 = to_orig(bb[:4])
    d.rectangle([x1, y1, x2, y2], outline=col, width=3)
    lbl = f"{names.get(str(cls), '?')} {bb[4]:.2f}"
    tw = d.textlength(lbl) + 6
    d.rectangle([x1, max(0.0, y1 - 16), x1 + tw, y1], fill=col)
    d.text((x1 + 3, max(0.0, y1 - 14)), lbl, fill=(255, 255, 255))

out = Image.alpha_composite(img0.convert("RGBA"), overlay).convert("RGB")
out.save(OUT)
print("saved", OUT)
