# YOLO26-pose truth generator. Reuses gen_ref.py (import) for RefWriter,
# fold_conv_bn, perturb_bns, export_folds, letterbox_image, capture hooks.
# Builds PoseModel on yolo26-pose.yaml (random perturbed) or loads official
# yolo26n-pose.pt (--weights). Backbone 0..22 is identical to detection, so the
# P4 full-graph builder applies unchanged; layer 23 = Pose26 adds one2one_cv4
# (85 ch) -> one2one_cv4_kpts (1x1 -> 51 ch) per level. Dumps per-layer io,
# raw heads (box/cls x3 + kpt x3), decoded detect_out [1,k,6+51], fold table,
# folded attention weights, names.json.
import argparse, importlib.metadata as md, json
from pathlib import Path
import numpy as np
import torch

import gen_ref as gr

POSE_CFG = gr.ULTRA / "ultralytics" / "cfg" / "models" / "26" / "yolo26-pose.yaml"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="320,320")
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ref_pose"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    height, width = (int(v) for v in args.size.split(","))

    from ultralytics.nn.tasks import PoseModel
    torch.manual_seed(args.seed)
    if args.weights:
        _o = md.version
        md.version = lambda n: (_o(n) if n != "torchvision" else "0.0.0")
        ck = torch.load(str(args.weights), map_location="cpu", weights_only=False)
        model = ck["model"].float().eval()
    else:
        model = PoseModel(str(POSE_CFG), ch=3, nc=args.nc, verbose=False).eval()
        gr.perturb_bns(model)
    detect = model.model[-1]
    print(f"head: {type(detect).__name__} nc={getattr(detect,'nc',None)} "
          f"kpt_shape={getattr(detect,'kpt_shape',None)} nk={getattr(detect,'nk',None)} "
          f"nl={getattr(detect,'nl',None)} end2end={getattr(detect,'end2end',None)}")

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

    # raw head capture: last module of each one2one chain per level
    hcapt = {}
    def mk(name):
        def hook(module, inp, out):
            hcapt.setdefault(name, []).append(out)
        return hook
    def lastmod(ch):
        return ch[-1] if hasattr(ch, "__getitem__") else ch
    heads = {"box": detect.one2one_cv2, "cls": detect.one2one_cv3,
             "kpt": detect.one2one_cv4_kpts}
    for kind, chains in heads.items():
        for lvl, chain in enumerate(chains):
            handles.append(lastmod(chain).register_forward_hook(mk(f"head_{kind}{lvl}")))

    if args.image:
        x = gr.letterbox_image(args.image, height, width)
    else:
        x = torch.randn(1, 3, height, width)
    with torch.no_grad():
        y = model(x)
    for h in handles:
        h.remove()

    # pick the decoded [1,k,6+nk] tensor
    flat = []
    def walk(o):
        if torch.is_tensor(o):
            flat.append(o)
        elif isinstance(o, (list, tuple)):
            for t in o:
                walk(t)
    walk(y)
    decoded = None
    for t in flat:
        s = tuple(t.shape)
        if len(s) == 3 and s[0] == 1 and s[2] > 6:
            decoded = t
    if decoded is not None:
        writer.dump("detect_out", decoded, "pose decoded [1,k,6+nk]")
        print("detect_out", tuple(decoded.shape))
    else:
        print("WARN decode pick failed:", [tuple(t.shape) for t in flat])
    for nm, v in hcapt.items():
        assert len(v) == 1, nm
        writer.dump(nm, v[0], f"raw head {nm}")

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

    # fold table (inference path incl. pose one2one branches; attention convs
    # folded separately below) + folded attention weights
    folded_attn = gr.export_folds(writer, model, len(model.model) - 1, attentions)
    for order, attn in enumerate(attentions):
        f = folded_attn.get(f"attn{order}")
        if f is None:
            continue
        for tag in ("qkv", "pe", "proj"):
            w, b = f[tag]
            writer.dump(f"attn{order}_{tag}_w", w, f"attn{order} {tag} folded w")
            writer.dump(f"attn{order}_{tag}_b", b, f"attn{order} {tag} folded b")

    names = getattr(model, "names", None)
    if names:
        writer.dir.joinpath("names.json").write_text(
            json.dumps({int(k): str(v) for k, v in names.items()}, indent=2) + "\n")
    (writer.dir / "modules.json").write_text(
        json.dumps(gr.module_table(model, detect), indent=2) + "\n")
    print(f"wrote {writer.count} tensors to {writer.dir}")

if __name__ == "__main__":
    main()
