# YOLO26-seg truth generator. Reuses gen_ref.py (import) for RefWriter,
# fold_conv_bn, perturb_bns, export_folds, letterbox_image, capture hooks.
# Builds SegmentationModel on yolo26-seg.yaml (random-init + perturbed BN, or
# official yolo26n-seg.pt via --weights), runs one forward on a letterboxed
# image (--image) or randn, and dumps per-layer io (0..22, same topology as
# detection so the P4 full-graph builder applies unchanged), the 9 raw heads
# (box/cls/mask x 3 levels), the final decoded [1,k,6+nm] rows, the Proto26
# output [1,nm,Hp,Wp], and the fold table + folded attention weights.
import argparse, importlib.metadata as md, json
from pathlib import Path
import numpy as np
import torch

import gen_ref as gr

SEG_CFG = gr.ULTRA / "ultralytics" / "cfg" / "models" / "26" / "yolo26-seg.yaml"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="320,320")
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ref_seg"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    height, width = (int(v) for v in args.size.split(","))

    from ultralytics.nn.tasks import SegmentationModel
    torch.manual_seed(args.seed)
    if args.weights:
        _o = md.version
        md.version = lambda n: (_o(n) if n != "torchvision" else "0.0.0")
        ck = torch.load(str(args.weights), map_location="cpu", weights_only=False)
        model = ck["model"].float().eval()
    else:
        model = SegmentationModel(str(SEG_CFG), ch=3, nc=args.nc, verbose=False).eval()
        gr.perturb_bns(model)
    detect = model.model[-1]
    print(f"head: {type(detect).__name__} nc={getattr(detect,'nc',None)} "
          f"nm={getattr(detect,'nm',None)} npr={getattr(detect,'npr',None)} nl={getattr(detect,'nl',None)}")

    # ---- attention instances (2 in yolo26n backbone) -> folded weight dumps
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

    # raw head capture: last conv of each one2one chain per level
    hcapt = {}
    def mk(name):
        def hook(module, inp, out):
            hcapt.setdefault(name, []).append(out)
        return hook
    heads = {"box": detect.one2one_cv2, "cls": detect.one2one_cv3, "mask": detect.one2one_cv4}
    for kind, chains in heads.items():
        for lvl, chain in enumerate(chains):
            last = chain[-1] if hasattr(chain, "__getitem__") and len(chain) else chain
            handles.append(last.register_forward_hook(mk(f"head_{kind}{lvl}")))

    # proto output hook
    proto_mod = detect.proto
    handles.append(proto_mod.register_forward_hook(mk("proto_out")))

    # input
    if args.image:
        x = gr.letterbox_image(args.image, height, width)
    else:
        x = torch.randn(1, 3, height, width)
    with torch.no_grad():
        y = model(x)
    for h in handles:
        h.remove()

    # ---- pick decoded + proto from the seg forward return
    flat = []
    def walk(o):
        if torch.is_tensor(o):
            flat.append(o)
        elif isinstance(o, (list, tuple)):
            for t in o:
                walk(t)
    walk(y)
    decoded = proto = None
    for t in flat:
        s = tuple(t.shape)
        if len(s) == 3 and s[0] == 1 and s[2] > 6:
            decoded = t
        if len(s) == 4 and s[1] == getattr(detect, "nm", 32):
            proto = t
    if decoded is None or proto is None:
        print("WARN decode/proto pick failed; shapes:", [tuple(t.shape) for t in flat])
    if decoded is not None:
        writer.dump("detect_out", decoded, "seg decoded [1,k,6+nm]")
        print("detect_out", tuple(decoded.shape))
    if proto is not None:
        writer.dump("proto", proto, "Proto26 output [1,nm,Hp,Wp]")
        print("proto", tuple(proto.shape))
        # Mask-reconstruction reference for the first 8 decoded rows (they are
        # top-k ranked): mask = sigmoid(sum_nm proto[0,nm] * coef_row[nm]) on
        # the proto grid [Hp,Wp]. This is exactly what the engine-side C code
        # must reproduce from the (matched) coef row + proto dump.
        if decoded is not None:
            coefs = decoded[0, :8, 6:6 + proto.shape[1]].float()   # [8,nm]
            pv = proto[0].float()                                   # [nm,Hp,Wp]
            mask = torch.sigmoid(torch.einsum("rn,nhw->rhw", coefs, pv))  # [8,Hp,Wp]
            for i in range(mask.shape[0]):
                writer.dump(f"mask_gt{i}", mask[i:i+1], f"rebuilt mask row {i} (proto grid)")
            print("mask_gt0..%d dumped" % (mask.shape[0]-1))
    for nm2, v in hcapt.items():
        assert len(v) == 1, nm2
        writer.dump(nm2, v[0], f"raw head {nm2}")

    # per-layer io (0..22) + input
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

    # ---- fold table over inference path (backbone+neck+one2one heads);
    # Proto26 convs are Conv wrappers under model.23.proto.* -> excluded by the
    # one2one gate, and its ConvTranspose2d is not a Conv2d -> also skipped;
    # dump the proto subgraph explicitly below.
    folded_attn = gr.export_folds(writer, model, len(model.model) - 1, attentions)
    # dump folded attention weights (same attn<i>_* names as detection)
    for order, attn in enumerate(attentions):
        f = folded_attn.get(f"attn{order}")
        if f is None:
            continue
        for tag in ("qkv", "pe", "proj"):
            w, b = f[tag]
            writer.dump(f"attn{order}_{tag}_w", w, f"attn{order} {tag} folded w")
            writer.dump(f"attn{order}_{tag}_b", b, f"attn{order} {tag} folded b")

    # ---- Proto26 subgraph explicit weights (BN-folded Conv wrappers +
    # ConvTranspose2d which keeps [c_in, c_out, kh, kw] layout)
    p = proto_mod
    for tag, m2 in (("feat_refine0", p.feat_refine[0]), ("feat_refine1", p.feat_refine[1]),
                    ("feat_fuse", p.feat_fuse), ("cv1", p.cv1), ("cv2", p.cv2), ("cv3", p.cv3)):
        w, b = gr.fold_conv_bn(m2.conv, m2.bn) if hasattr(m2, "bn") else (m2.conv.weight, m2.conv.bias)
        writer.dump(f"proto_{tag}_w", w, f"proto {tag} folded w")
        writer.dump(f"proto_{tag}_b", b if b is not None else torch.zeros(m2.conv.out_channels),
                    f"proto {tag} folded b")
    u = p.upsample
    writer.dump("proto_upsample_w", u.weight, "ConvTranspose2d weight [c_in,c_out,2,2]")
    writer.dump("proto_upsample_b", u.bias if u.bias is not None else torch.zeros(u.out_channels),
                "ConvTranspose2d bias")

    names = getattr(model, "names", None)
    if names:
        writer.dir.joinpath("names.json").write_text(
            json.dumps({int(k): str(v) for k, v in names.items()}, indent=2) + "\n")
    (writer.dir / "modules.json").write_text(
        json.dumps(gr.module_table(model, detect), indent=2) + "\n")
    print(f"wrote {writer.count} tensors to {writer.dir}")

if __name__ == "__main__":
    main()
