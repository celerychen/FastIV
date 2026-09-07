import importlib.metadata as md, sys, types, time, json, torch, numpy as np, os
from PIL import Image

_o = md.version
md.version = lambda n: (_o(n) if n != "torchvision" else "0.0.0")

class _StubBase:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
    def __call__(self, *a, **k): return a[0] if a else None
class _Fuzzy(types.ModuleType):
    _cache = {}
    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = type(name, (_StubBase,), {})
        return self._cache[name]
def stub_tv():
    for name in ("torchvision", "torchvision.transforms", "torchvision.transforms.transforms",
                 "torchvision.transforms.functional", "torchvision.transforms.functional_tensor"):
        sys.modules.setdefault(name, _Fuzzy(name))
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
    sys.modules["torchvision.transforms"].transforms = sys.modules["torchvision.transforms.transforms"]
    sys.modules["torchvision.transforms"].functional = sys.modules["torchvision.transforms.functional"]

MODELS = "/Users/celerychen2024/hqai/FastIV/app/yolo26/models"
IMG = "/Users/celerychen2024/hqai/FastIV/app/yolo26/test/real/bus.jpg"
OUT = "/tmp/t_e2e"

img0 = Image.open(IMG).convert("RGB")

def letterbox_from(img, size):
    iw, ih = img.size
    r = min(size/iw, size/ih)
    nw, nh = max(1, round(iw*r)), max(1, round(ih*r))
    im = img.resize((nw, nh))
    cv = Image.new("RGB", (size, size), (114, 114, 114))
    cv.paste(im, ((size-nw)//2, (size-nh)//2))
    return torch.from_numpy(np.asarray(cv, dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0)

def walk_tensors(o, acc):
    if torch.is_tensor(o): acc.append(o)
    elif isinstance(o, (list, tuple)):
        for t in o: walk_tensors(t, acc)

def bench_and_dump(name, pt, task):
    stub_tv()
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    m = ck["model"].float().eval()
    names = ck.get("names") or getattr(m, "names", None)
    x = letterbox_from(img0, 320)
    warmup, runs = 5, 20
    with torch.no_grad():
        for _ in range(warmup): m(x)
        fwd = []
        for _ in range(runs):
            t0 = time.perf_counter(); m(x)
            fwd.append((time.perf_counter()-t0)*1e3)
        e2e = []
        for _ in range(runs):
            t0 = time.perf_counter()
            xs = letterbox_from(img0, 320)
            m(xs)
            e2e.append((time.perf_counter()-t0)*1e3)
    print(f"{name:10s} torch forward best {min(fwd):7.2f} ms avg {sum(fwd)/runs:7.2f} ms | "
          f"+letterbox best {min(e2e):7.2f} ms avg {sum(e2e)/runs:7.2f} ms")

    with torch.no_grad():
        y = m(x)
    tensors = []
    walk_tensors(y, tensors)
    dec = None; proto = None; probs = None; logits = None
    for t in tensors:
        s = tuple(t.shape)
        if len(s) == 3 and s[0] == 1 and 1 < s[1] < 10000 and s[2] in (6, 57, 38) and dec is None:
            dec = t
        if len(s) == 4 and s[1] == 32 and proto is None:
            proto = t
        if len(s) == 2 and s[1] == 1000:
            v = t[0].float()
            if float(v.min()) >= 0.0 and float(v.max()) <= 1.0:
                probs = t   # softmax output lives in [0,1]
            else:
                logits = t
    if task == "detector" and dec is not None:
        d = dec[0].float().cpu().numpy()
        with open(f"{OUT}/detector.txt", "w") as f:
            for row in d:
                f.write("%.3f %.3f %.3f %.3f %.6f %d\n" % (row[0],row[1],row[2],row[3],row[4],int(round(row[5]))))
    elif task == "segmenter" and dec is not None and proto is not None:
        d = dec[0].float().cpu().numpy()
        p = proto[0].float().cpu().numpy()
        with open(f"{OUT}/segmenter.txt", "w") as f:
            for row in d:
                f.write("%.3f %.3f %.3f %.3f %.6f %d" % (row[0],row[1],row[2],row[3],row[4],int(round(row[5]))))
                for m in range(32): f.write(" %.6f" % row[6+m])
                f.write("\n")
        p.tofile(f"{OUT}/seg_proto.bin")
    elif task == "pose" and dec is not None:
        d = dec[0].float().cpu().numpy()
        with open(f"{OUT}/pose.txt", "w") as f:
            for row in d:
                f.write("%.3f %.3f %.3f %.3f %.6f %d" % (row[0],row[1],row[2],row[3],row[4],int(round(row[5]))))
                for k in range(51): f.write(" %.3f" % row[6+k])
                f.write("\n")
    elif task == "classifier" and probs is not None:
        pr = probs[0].float().cpu().numpy()
        order = np.argsort(-pr)[:5]
        with open(f"{OUT}/classifier.txt", "w") as f:
            for c in order: f.write("%d %.6f\n" % (int(c), float(pr[c])))
    if names and task != "classifier":
        try:
            top = np.argsort(-dec[0,:,4].float().cpu().numpy())
            print(f"    top detections: " + ", ".join(
                f"{names.get(str(int(round(dec[0,i,5].item()))),'?')} {dec[0,i,4].item():.2f}"
                for i in top[:4]))
        except Exception: pass
    elif task == "classifier":
        pr = probs[0].float().cpu().numpy()
        order = np.argsort(-pr)[:5]
        nm = names if names else {}
        print("    top5: " + ", ".join(f"{nm.get(str(int(c)),'?')} {pr[c]:.3f}" for c in order))

bench_and_dump("detector",  f"{MODELS}/yolo26n.pt",      "detector")
bench_and_dump("segmenter", f"{MODELS}/yolo26n-seg.pt",  "segmenter")
bench_and_dump("pose",      f"{MODELS}/yolo26n-pose.pt", "pose")
bench_and_dump("classifier",f"{MODELS}/yolo26n-cls.pt",  "classifier")
print("dumped to", OUT)
