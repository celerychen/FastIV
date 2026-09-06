# YOLO26-cls truth generator. Backbone (layers 0..9) == detection backbone up to
# C2PSA, layer10 = Conv(256->1280) + AdaptiveAvgPool2d(1) + Dropout + Linear.
# The whole head after the Conv is pool+linear -> a plain C post-process, so we
# dump: per-layer io 0..10, the raw pooled feature is NOT needed; the engine side
# builds layers 0..10 (Conv acts handled by fold), then C code does global-avg
# pool + Linear -> logits, compared against torch logits (avgpool+linear are
# deterministic fp ops, exact match expected).
import argparse, importlib.metadata as md, json, sys, types
from pathlib import Path
import numpy as np
import torch

import gen_ref as gr

CLS_CFG = gr.ULTRA / "ultralytics" / "cfg" / "models" / "26" / "yolo26-cls.yaml"

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

def stub_torchvision():
    for name in ("torchvision", "torchvision.transforms", "torchvision.transforms.transforms",
                 "torchvision.transforms.functional", "torchvision.transforms.functional_tensor"):
        sys.modules.setdefault(name, _Fuzzy(name))
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
    sys.modules["torchvision.transforms"].transforms = sys.modules["torchvision.transforms.transforms"]
    sys.modules["torchvision.transforms"].functional = sys.modules["torchvision.transforms.functional"]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="320,320")
    ap.add_argument("--nc", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ref_cls"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    height, width = (int(v) for v in args.size.split(","))

    from ultralytics.nn.tasks import ClassificationModel
    torch.manual_seed(args.seed)
    if args.weights:
        _o = md.version
        md.version = lambda n: (_o(n) if n != "torchvision" else "0.0.0")
        stub_torchvision()
        ck = torch.load(str(args.weights), map_location="cpu", weights_only=False)
        model = ck["model"].float().eval()
    else:
        model = ClassificationModel(str(CLS_CFG), ch=3, nc=args.nc, verbose=False).eval()
        gr.perturb_bns(model)
    head = model.model[-1]
    print(f"head: {type(head).__name__}  nc(out)={getattr(head,'nc',None)}  net_layers={len(model.model)}")

    attentions = [m for m in model.modules() if type(m).__name__ == "Attention"]
    writer = gr.RefWriter(args.out)

    captures, handles, layer_keys = {}, [], {}
    for index, module in enumerate(model.model):
        if index == len(model.model) - 1:
            continue
        key = f"layer{index:02d}_{type(module).__name__}"
        layer_keys[index] = key
        handles.append(module.register_forward_hook(gr.capture_output(captures, key)))
        handles.append(module.register_forward_hook(gr.capture_input(captures, key + "_in")))

    # raw post-Conv feature (input to AdaptiveAvgPool) capture on the head conv
    hcapt = {}
    def mk(name):
        def hook(module, inp, out):
            hcapt.setdefault(name, []).append(out)
        return hook
    head_conv = head.conv  # ultralytics Classify: self.conv = Conv(...)
    handles.append(head_conv.register_forward_hook(mk("head_conv_out")))

    if args.image:
        x = gr.letterbox_image(args.image, height, width)
    else:
        x = torch.randn(1, 3, height, width)
    with torch.no_grad():
        y = model(x)   # list [(1,nc),(1,nc)]: logits then probs (eval)
    for h in handles:
        h.remove()

    # ClassificationModel eval forward returns (probs, logits) - y[0] is the
    # softmax output (sums to 1), y[1] the raw logits.
    if isinstance(y, (list, tuple)):
        probs = y[0]
        logits = y[1] if len(y) > 1 else None
    else:
        logits = y
        probs = None
    writer.dump("logits", logits, "cls logits [1,nc]")
    if probs is not None:
        writer.dump("probs", probs, "cls softmax probs [1,nc]")
    print("logits", tuple(logits.shape), "range", float(logits.min()), float(logits.max()))

    # pooled feature + linear weight/bias for the C post-process reference
    v = hcapt.get("head_conv_out")
    if v and len(v) == 1:
        feat = v[0]
        # adaptive avg pool (1) == mean over spatial
        pooled = feat.mean(dim=(2, 3))           # [1,1280]
        writer.dump("head_pooled", pooled, "post head-conv global-avg pooled [1,1280]")
        print("pooled", tuple(pooled.shape))
    # head conv (Conv 1x1 256->1280, SiLU, BN) is model.10 and NOT in the fold
    # table (export_folds gates at layer<detect_index); dump its BN-folded weights
    # so the C engine can add the head conv node itself.
    hw, hb = gr.fold_conv_bn(head.conv.conv, head.conv.bn)
    writer.dump("head_conv_w", hw, "cls head Conv BN-folded weight [1280,256,1,1]")
    writer.dump("head_conv_b", hb, "cls head Conv BN-folded bias [1280]")
    print("head_conv", tuple(hw.shape), tuple(hb.shape))
    lw = head.linear.weight
    lb = head.linear.bias
    if lb is None:
        lb = torch.zeros(lw.shape[0])
    writer.dump("linear_w", lw, "head Linear weight [nc,1280]")
    writer.dump("linear_b", lb, "head Linear bias [nc]")
    print("linear", tuple(lw.shape))

    writer.dump("input", x, "network input")
    for index in sorted(layer_keys):
        key = layer_keys[index]
        v = captures.get(key)
        if v and len(v) == 1:
            writer.dump(key, v[0], f"out model.model[{index}]")
        ik = key + "_in"
        v = captures.get(ik)
        if v and len(v) == 1:
            writer.dump(ik, v[0], f"in model.model[{index}]")

    # fold table over layers 0..10 (Conv wrappers only; linear has no BN)
    folded_attn = gr.export_folds(writer, model, len(model.model) - 1, attentions)
    for order, attn in enumerate(attentions):
        f = folded_attn.get(f"attn{order}")
        if f is None:
            continue
        for tag in ("qkv", "pe", "proj"):
            w, b = f[tag]
            writer.dump(f"attn{order}_{tag}_w", w, f"attn{order} {tag} folded w")
            writer.dump(f"attn{order}_{tag}_b", b, f"attn{order} {tag} folded b")

    names = getattr(model, "names", None) or {}
    if names:
        writer.dir.joinpath("names.json").write_text(
            json.dumps({int(k): str(v) for k, v in names.items()}, indent=2) + "\n")
    (writer.dir / "modules.json").write_text(
        json.dumps(gr.module_table(model, head), indent=2) + "\n")
    print(f"wrote {writer.count} tensors to {writer.dir}")

if __name__ == "__main__":
    main()
