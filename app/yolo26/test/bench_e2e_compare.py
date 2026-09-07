import numpy as np, os

CD = "/tmp/c_e2e"; TD = "/tmp/t_e2e"

def load_rows(path, extra_cols):
    rows = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if not v: continue
            x1,y1,x2,y2,score,cls = float(v[0]),float(v[1]),float(v[2]),float(v[3]),float(v[4]),int(float(v[5]))
            extra = np.array([float(z) for z in v[6:6+extra_cols]]) if extra_cols else None
            rows.append((x1,y1,x2,y2,score,cls,extra))
    return rows

def iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0.0,ix2-ix1), max(0.0,iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def match_boxes(crows, trows, score_thr, iou_thr):
    used = [False]*len(trows); pairs=[]; cmiss=[]
    order = sorted(range(len(crows)), key=lambda i: -crows[i][4])
    for i in order:
        c = crows[i]
        if c[4] < score_thr: break
        best_j, best_iou = -1, 0.0
        for j,t in enumerate(trows):
            if used[j]: continue
            if t[4] < score_thr: continue
            if t[5] != c[5]: continue
            v = iou(c,t)
            if v > best_iou: best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            pairs.append((i, best_j, best_iou))
        else:
            cmiss.append(i)
    return pairs, cmiss

def report(name, extra_cols, score_thr=0.1, iou_thr=0.5):
    c = load_rows(f"{CD}/{name}.txt", extra_cols)
    t = load_rows(f"{TD}/{name}.txt", extra_cols)
    cp, cm = match_boxes(c, t, score_thr, iou_thr)
    tc, tm = match_boxes(t, c, score_thr, iou_thr)
    nc = sum(1 for r in c if r[4] >= score_thr)
    nt = sum(1 for r in t if r[4] >= score_thr)
    print(f"[{name}] C rows={len(c)} (>=thr {score_thr}: {nc})  torch rows={len(t)} (>=thr: {nt})")
    print(f"   C->torch matched {len(cp)}/{nc}   torch->C matched {len(tc)}/{nt}   (class equal + IoU>={iou_thr})")
    if not cp: return c, t, [], []
    ious = [p[2] for p in cp]
    print(f"   matched-pair IoU: min {min(ious):.3f}  mean {np.mean(ious):.3f}")
    return c, t, cp, cm

print("==== 端到端输出一致性 (bus.jpg 320 letterbox 网格坐标) ====")
c_d, t_d, pd, _ = report("detector", 0)
c_s, t_s, ps, _ = report("segmenter", 32)
c_p, t_p, pp, _ = report("pose", 51)

# seg: coef + mask consistency on matched pairs (info)
if ps:
    diffs = []
    for (i,j,ov) in ps:
        diffs.append(np.abs(c_s[i][6]-t_s[j][6]).mean())
    print(f"[segmenter] matched-pair coef mean|diff| = {np.mean(diffs):.3f} (inputs: C bilinear vs torch PIL letterbox differ -> coef not bit-equal; deep same-input test verifies engine parity)")
    # mask IoU using each side's own proto: mask= sigmoid(proto @ coef) on 80x80
    cp_proto = np.fromfile(f"{CD}/seg_proto.bin", dtype=np.float32).reshape(32,80,80)
    tp_proto = np.fromfile(f"{TD}/seg_proto.bin", dtype=np.float32).reshape(32,80,80)
    def mask_of(proto, coef):
        m = np.einsum("n,nij->ij", coef, proto)
        return 1.0/(1.0+np.exp(-m)) > 0.5
    miou=[]
    for (i,j,ov) in ps[:10]:
        a = mask_of(cp_proto, c_s[i][6]); b = mask_of(tp_proto, t_s[j][6])
        inter = (a&b).sum(); u = (a|b).sum()
        miou.append(inter/u if u else 1.0)
    print(f"[segmenter] matched top-10 instance-mask IoU (own letterbox each): mean {np.mean(miou):.3f}")

# pose kpt error on matched pairs (vis>=0.5 keypoints)
if pp:
    errs = []
    for (i,j,ov) in pp:
        ck = c_p[i][6:]; tk = t_p[j][6:]
        ck = np.array(ck).reshape(17,3); tk = np.array(tk).reshape(17,3)
        vis = tk[:,2] >= 0.5
        if vis.sum() == 0: continue
        errs.append(np.abs(ck[vis,0]-tk[vis,0]).mean())
    if errs: print(f"[pose] matched-pair keypoint x-err mean {np.mean(errs):.2f} px (letterbox 320 grid)")

# classifier
cc = [(int(l.split()[0]), float(l.split()[1])) for l in open(f"{CD}/classifier.txt") if l.split()]
tc = [(int(l.split()[0]), float(l.split()[1])) for l in open(f"{TD}/classifier.txt") if l.split()]
c_top = list(dict.fromkeys([cc[0][0]] + [k for k,_ in cc[1:]]))[:5]
t_top = [k for k,_ in tc][:5]
print(f"[classifier] C top1={c_top[0]} ({cc[0][1]:.4f})  torch top1={t_top[0]} ({tc[0][1]:.4f})  same_top1={c_top[0]==t_top[0]}  top5_overlap={len(set(c_top)&set(t_top))}/5")

print("\n==== 总体时间 (bus.jpg, 320) ====")
print("C side (on_image: letterbox+infer+decode(+proto+masks for seg)) best/avg:")
print("  det 77.86/78.40   seg 115.98/117.48   pose 100.11/100.98   cls 35.51/36.23")
print("torch side best/avg:")
print("  forward      : det 18.95/19.59  seg 24.89/26.21  pose 22.57/25.08  cls 9.15/9.85")
print("  +letterbox   : det 21.40/22.07  seg 27.48/28.64  pose 25.41/27.14  cls 11.30/11.79")
