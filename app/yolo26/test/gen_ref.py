#!/usr/bin/env python3
"""Generate torch ground-truth tensors for the YOLO26 -> FastIV port.

Usage (from anywhere; paths are resolved relative to this file):

    python3 app/yolo26/test/gen_ref.py --size 64,64 --nc 80

Writes into app/yolo26/test/ref/:

    <name>.f32      raw float32 little-endian array, no header
    <name>.json     {"name", "shape", "dtype", "src"}
    weights.f32     every float32 parameter/buffer, concatenated
    weights.json    [{"name", "shape", "offset", "count"}, ...]
    modules.json    top-level module table (index, type, key attributes)
    attention.json  per-instance Attention hyper-parameters
    MD5SUMS.txt     md5 of every generated file, for determinism checks

Groups dumped:
    input             the exact tensor fed to the network
    layerNN_<Type>    output of model.model[NN] (24 modules, Detect excluded)
    layer09_sppf_y*   SPPF internals: cv1 output + the 3 chained max-pool stages
    attn{0,1}_*       Attention stepwise intermediates (qkv/q/k/v/S/A/O/...)
    head_box{i}       detect.one2one_cv2[i] raw output (undecoded)
    head_cls{i}       detect.one2one_cv3[i] raw output (undecoded)
    boxes_cat/scores_cat/scores_sig/dbox/anchors/strides/infer_cat/detect_out
    syn_*             synthetic tensors used by the P1 operator unit tests
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ULTRA = REPO / "src" / "reference" / "ultralytics-main"
CFG = ULTRA / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ULTRA))

import numpy as np
import torch  # noqa: E402

import torch_env  # noqa: E402,F401  metadata shim, must precede ultralytics


class RefWriter:
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def dump(self, name, tensor, src):
        if tensor is None:
            raise ValueError(f"{name}: tensor is None")
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().to(torch.float32).contiguous()
        else:
            # None bias -> emit a zero vector of the requested length
            array = torch.zeros(int(tensor), dtype=torch.float32).contiguous()
        (self.dir / f"{name}.f32").write_bytes(array.numpy().tobytes())
        meta = {"name": name, "shape": list(array.shape), "dtype": "f32", "src": src}
        (self.dir / f"{name}.json").write_text(json.dumps(meta) + "\n")
        self.count += 1

    def dump_bias(self, name, conv, out_ch, src):
        # ultralytics Conv uses bias=False -> dump zeros when None
        b = conv.bias if conv.bias is not None else torch.zeros(out_ch)
        self.dump(name, b, src)


def capture_output(store, name):
    def hook(module, inputs, output):
        store.setdefault(name, []).append(output)

    return hook


def capture_input(store, name):
    def hook(module, inputs, output):
        # Top-level Concat receives a tuple/list of tensors (no single input);
        # skip those - the block-input dump is for single-tensor blocks only.
        if inputs and isinstance(inputs[0], torch.Tensor):
            store.setdefault(name, []).append(inputs[0])

    return hook


def attention_stepwise(attn, x):
    """Re-run Attention.forward, keeping every intermediate tensor.

    Mirrors ultralytics/nn/modules/block.py Attention.forward exactly; the caller
    asserts the result is bit-identical to the real forward, which proves the
    dumped intermediates really come from the executed compute path.
    """
    batch, channels, height, width = x.shape
    num_tokens = height * width
    qkv = attn.qkv(x)
    query, key, value = qkv.view(batch, attn.num_heads, attn.key_dim * 2 + attn.head_dim, num_tokens).split(
        [attn.key_dim, attn.key_dim, attn.head_dim], dim=2
    )
    scores = (query * attn.scale).transpose(-2, -1) @ key
    weights = scores.softmax(dim=-1)
    context = value @ weights.transpose(-2, -1)
    attn_out = context.view(batch, channels, height, width)
    pe_out = attn.pe(value.reshape(batch, channels, height, width))
    out = attn.proj(attn_out + pe_out)
    return out, {
        "qkv": qkv,
        "q": query,
        "k": key,
        "v": value,
        "S": scores,
        "A": weights,
        "O": context,
        "attn_out": attn_out,
        "pe_out": pe_out,
        "out": out,
    }


def module_table(model, detect):
    rows = []
    for index, module in enumerate(model.model):
        kind = type(module).__name__
        row = {"index": index, "type": kind}
        if kind == "Conv":
            row.update(
                in_c=module.conv.in_channels,
                out_c=module.conv.out_channels,
                k=module.conv.kernel_size[0],
                s=module.conv.stride[0],
                g=module.conv.groups,
                act=type(module.act).__name__,
            )
        elif kind == "C3k2":
            row.update(
                sub=type(module.m[0]).__name__,
                n=len(module.m),
                cv1=module.cv1.conv.out_channels,
                cv2=module.cv2.conv.out_channels,
            )
        elif kind == "SPPF":
            row.update(
                k=module.m.kernel_size,
                s=module.m.stride,
                p=module.m.padding,
                add=bool(getattr(module, "add", False)),
                cv1=module.cv1.conv.out_channels,
                cv2=module.cv2.conv.out_channels,
            )
        elif kind == "C2PSA":
            row.update(c=module.c, n=len(module.m))
        elif kind == "Detect":
            row.update(
                nl=detect.nl,
                nc=detect.nc,
                reg_max=detect.reg_max,
                end2end=bool(detect.end2end),
                strides=[float(v) for v in detect.stride],
            )
        rows.append(row)
    return rows


def perturb_bns(root, base_seed=20260906):
    """Deterministically pull every BatchNorm2d of `root` away from its identity
    init (gamma=1, beta=0, running_mean=0, running_var=1). With random-init convs
    an untouched BN makes the fold formula degenerate (w' == w at 1e-5), so fold
    bugs (missing mean term, missing sqrt) would be invisible to float tolerance.
    Each BN gets its own generator so the perturbation is stable regardless of
    how many random draws the model construction consumed. Call BEFORE the first
    forward so every dumped tensor reflects the perturbed (non-degenerate) BN."""
    for i, m in enumerate(root.modules()):
        if not isinstance(m, torch.nn.BatchNorm2d):
            continue
        g = torch.Generator().manual_seed(base_seed + i)
        n = m.num_features
        m.weight.data.copy_(torch.rand(n, generator=g) * 0.8 + 0.6)   # gamma in [0.6, 1.4]
        m.bias.data.copy_(torch.rand(n, generator=g) * 0.6 - 0.3)     # beta  in [-0.3, 0.3]
        m.running_mean.copy_(torch.rand(n, generator=g) - 0.5)        #       in [-0.5, 0.5]
        m.running_var.copy_(torch.rand(n, generator=g) * 1.5 + 0.5)   # var   in [0.5, 2.0]


def fold_conv_bn(conv, bn):
    """Fold BatchNorm into an adjacent Conv2d (bias=False):
         s[c]  = gamma[c] / sqrt(var[c] + eps)
         W'[co] = W[co] * s[co]
         b'[co] = beta[co] - s[co] * running_mean[co]
    Layout (C_out, C_in, ky, kx) is unchanged (identical to FastIV)."""
    scale = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    w = (conv.weight.data * scale.view(-1, 1, 1, 1)).contiguous()
    b = (bn.bias.data - scale * bn.running_mean.data).contiguous()
    return w, b


def _act_kind(conv_parent):
    """Map a Conv wrapper's activation to an int: 0 = none/Identity, 1 = SiLU."""
    act = getattr(conv_parent, "act", None)
    if act is None or type(act).__name__ == "Identity":
        return 0
    if type(act).__name__ == "SiLU":
        return 1
    raise ValueError(f"unhandled activation {type(act).__name__}")


def _on_inference_path(path, detect_index):
    """Keep only convs on the inference path: backbone+neck (model.model[<D]) and
    the Detect one2one branch. The one2many heads (Detect.cv2/cv3) are skipped."""
    parts = path.split(".")
    if len(parts) < 2 or parts[0] != "model":
        return False
    try:
        idx = int(parts[1])
    except ValueError:
        return False
    if idx < detect_index:
        return True
    if idx > detect_index:
        return False
    return "one2one" in path and ".cv2." not in path and ".cv3." not in path


def export_folds(writer, model, detect_index, attentions):
    """Fold every Conv2d+BN on the inference path (plus the Attention-internal
    qkv/pe/proj convs, returned separately for the P2 dump) and write:
      fold_w.f32 / fold_b.f32   concatenated folded weights/biases
      fold.json                 one JSON object per conv (see yolo26_ref.h)
    Self-check: for every folded conv, bn(conv(x)) == conv2d(x, W', b') on a
    fixed random input. Plain Conv2d without BN (e.g. Detect tail) is exported
    as-is. Returns {prefix: {"qkv":(w,b), "pe":(w,b), "proj":(w,b)}} for the
    Attention instances (used by main() to dump folded attn weights)."""
    import torch.nn.functional as Fn

    folded_by_prefix = {}
    for order, attn in enumerate(attentions):
        folded_by_prefix[f"attn{order}"] = {
            "qkv": fold_conv_bn(attn.qkv.conv, attn.qkv.bn),
            "pe": fold_conv_bn(attn.pe.conv, attn.pe.bn),
            "proj": fold_conv_bn(attn.proj.conv, attn.proj.bn),
        }

    entries = []
    w_payload = bytearray()
    b_payload = bytearray()
    seq = 0
    max_errs = []
    for path, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if not _on_inference_path(path, detect_index):
            continue
        if any(seg == "attn" for seg in path.split(".")):
            continue  # Attention convs are folded inside the composite node
        parent = None
        folded = None
        if path.endswith(".conv"):
            parent = model.get_submodule(path[: -len(".conv")])
            bn = getattr(parent, "bn", None)
            if isinstance(bn, torch.nn.BatchNorm2d):
                folded = fold_conv_bn(m, bn)
        has_bn = folded is not None
        w = folded[0] if folded else m.weight.data.contiguous()
        if folded:
            b = folded[1]
        else:
            b = m.bias.data if m.bias is not None else torch.zeros(m.out_channels)
        b = b.contiguous().to(torch.float32)
        w = w.contiguous().to(torch.float32)

        if has_bn:
            # self-check: bn(conv(x)) == conv2d(x, W', b')
            x = torch.randn(1, m.in_channels, 16, 16)
            with torch.no_grad():
                y_ref = bn(m(x))
                y_fold = Fn.conv2d(x, w, b, stride=m.stride, padding=m.padding, groups=m.groups)
            err = (y_ref - y_fold).abs().max().item()
            max_errs.append(err)

        w_off = len(w_payload) // 4
        w_payload += w.numpy().tobytes()
        b_off = len(b_payload) // 4
        b_payload += b.numpy().tobytes()
        entries.append(
            {
                "seq": seq,
                "layer": int(path.split(".")[1]),
                "in_c": m.in_channels,
                "out_c": m.out_channels,
                "k": m.kernel_size[0],
                "s": m.stride[0],
                "p": m.padding[0] if isinstance(m.padding, tuple) else int(m.padding),
                "g": m.groups,
                "has_bn": int(has_bn),
                "act": _act_kind(parent) if parent is not None else 0,
                "path": path,
                "parent": type(parent).__name__ if parent is not None else "-",
                "w_off": w_off,
                "w_cnt": w.numel(),
                "b_off": b_off,
                "b_cnt": b.numel(),
            }
        )
        seq += 1

    (writer.dir / "fold_w.f32").write_bytes(bytes(w_payload))
    (writer.dir / "fold_b.f32").write_bytes(bytes(b_payload))
    lines = [json.dumps(e, separators=(",", ":")) for e in entries]
    (writer.dir / "fold.json").write_text("\n".join(lines) + "\n")

    if max_errs:
        worst = max(max_errs)
        print(f"[check] BN fold self-check over {len(max_errs)} convs: worst max_abs_err={worst:g}")
        assert worst < 1e-4, "BN fold self-check failed"
    print(f"fold table: {seq} convs ({len(w_payload) // 4} w floats, {len(b_payload) // 4} b floats)")
    return folded_by_prefix


def dump_weights(writer, model):
    blobs = []
    payload = bytearray()
    seen = set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name in seen or tensor.dtype != torch.float32:
            continue
        array = tensor.detach().to(torch.float32).contiguous()
        blobs.append(
            {"name": name, "shape": list(array.shape), "offset": len(payload) // 4, "count": array.numel()}
        )
        payload += array.numpy().tobytes()
    (writer.dir / "weights.f32").write_bytes(bytes(payload))
    (writer.dir / "weights.json").write_text(json.dumps(blobs) + "\n")
    return len(blobs), len(payload) // 4


def dump_synthetic(writer):
    """Small fixed tensors for the P1 operator unit tests (SILU / MAXPOOL / SLICE)."""
    silu_src = torch.randn(1, 8, 5, 7)
    writer.dump("syn_silu_src", silu_src, "random tensor, P1 SILU input")
    writer.dump("syn_silu_out", torch.nn.functional.silu(silu_src), "F.silu(syn_silu_src)")

    pool_src = torch.randn(1, 4, 9, 11)
    writer.dump("syn_pool_src", pool_src, "random tensor, P1 MAXPOOL input")
    writer.dump("syn_pool_k5s1p2", torch.nn.functional.max_pool2d(pool_src, 5, 1, 2), "max_pool2d k=5 s=1 p=2")
    writer.dump("syn_pool_k3s2p1", torch.nn.functional.max_pool2d(pool_src, 3, 2, 1), "max_pool2d k=3 s=2 p=1")
    writer.dump("syn_pool_k2s2p0", torch.nn.functional.max_pool2d(pool_src, 2, 2, 0), "max_pool2d k=2 s=2 p=0")

    slice_src = torch.randn(1, 12, 4, 5)
    half0, half1 = slice_src.chunk(2, dim=1)
    writer.dump("syn_slice_src", slice_src, "random tensor, P1 SLICE input")
    writer.dump("syn_slice_half0", half0, "slice_src.chunk(2, dim=1)[0]")
    writer.dump("syn_slice_half1", half1, "slice_src.chunk(2, dim=1)[1]")

    split_src = torch.randn(1, 256, 4, 4)
    seg_q, seg_k, seg_v = split_src.view(1, 2, 32 + 32 + 64, 16).split([32, 32, 64], dim=2)
    writer.dump("syn_split_src", split_src, "random tensor shaped like a qkv output (heads=2)")
    writer.dump("syn_split_q", seg_q, "qkv.view(1,2,128,16).split([32,32,64],dim=2)[0]")
    writer.dump("syn_split_k", seg_k, "qkv.view(1,2,128,16).split([32,32,64],dim=2)[1]")
    writer.dump("syn_split_v", seg_v, "qkv.view(1,2,128,16).split([32,32,64],dim=2)[2]")

    # P2 synthetic ATTENTION case: the two real Attention instances in the 64x64
    # net operate on 2x2 feature maps (N=4) whose q/k/v are ~0, so softmax is a
    # uniform 0.25 regardless of scale and a q/k swap or a dropped pe term is
    # invisible to the test. Build a standalone Attention (same hyperparams:
    # dim=128, heads=2 -> head_dim=64, key_dim=32, scale=32**-0.5) on a 16x16
    # input (N=256) so the P2 test actually exercises the attention math.
    from ultralytics.nn.modules.block import Attention

    syn_attn = Attention(dim=128, num_heads=2, attn_ratio=0.5).eval()
    perturb_bns(syn_attn, base_seed=20260910)  # distinct base: non-degenerate BN
    syn_x = torch.randn(1, 128, 16, 16)
    with torch.no_grad():
        y_orig = syn_attn(syn_x)
        y_step, syn_inter = attention_stepwise(syn_attn, syn_x)
    err = (y_orig - y_step).abs().max().item()
    print(f"[check] syn_attn stepwise vs torch forward: max_abs_err={err:g}")
    assert err == 0.0, "synthetic stepwise dump does not follow the real compute path"
    writer.dump("syn_attn_in", syn_x, "synthetic Attention input x [1,128,16,16], N=256")
    for suffix in ("qkv", "q", "k", "v", "S", "A", "O", "attn_out", "pe_out", "out"):
        writer.dump(f"syn_attn_{suffix}", syn_inter[suffix], f"synthetic Attention intermediate '{suffix}'")
    # BN-folded weights so the composite-node test covers the BN path too.
    q_w, q_b = fold_conv_bn(syn_attn.qkv.conv, syn_attn.qkv.bn)
    p_w, p_b = fold_conv_bn(syn_attn.pe.conv, syn_attn.pe.bn)
    j_w, j_b = fold_conv_bn(syn_attn.proj.conv, syn_attn.proj.bn)
    writer.dump("syn_attn_qkv_w", q_w, "syn Attention qkv conv BN-folded weight [qkv_ch,dim,1,1]")
    writer.dump("syn_attn_qkv_b", q_b, "syn Attention qkv conv BN-folded bias [qkv_ch]")
    writer.dump("syn_attn_pe_w", p_w, "syn Attention pe dw conv BN-folded weight [dim,1,3,3]")
    writer.dump("syn_attn_pe_b", p_b, "syn Attention pe dw conv BN-folded bias [dim]")
    writer.dump("syn_attn_proj_w", j_w, "syn Attention proj conv BN-folded weight [dim,dim,1,1]")
    writer.dump("syn_attn_proj_b", j_b, "syn Attention proj conv BN-folded bias [dim]")


def letterbox_image(path, height, width):
    """Ultralytics-style letterbox: keep aspect ratio, pad to (height,width)
    with gray 114, then normalize to float32/255. Returns an RGB CHW tensor
    with a batch dim [1,3,H,W] -- the exact tensor the model must see so the
    C-side decode can be compared with torch detect_out on the same input."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    iw, ih = img.size
    r = min(width / iw, height / ih)
    nw, nh = max(1, round(iw * r)), max(1, round(ih * r))
    img = img.resize((nw, nh))
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(img, ((width - nw) // 2, (height - nh) // 2))
    a = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)  # [H,W,3] RGB
    a = a.permute(2, 0, 1).unsqueeze(0).contiguous()                    # [1,3,H,W]
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="64,64", help="input height,width (default 64,64)")
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(HERE / "ref"))
    parser.add_argument("--weights", default=None,
                        help="load a pretrained .pt instead of random-init (skips BN perturbation)")
    parser.add_argument("--image", default=None,
                        help="letterbox this image as model input instead of randn")
    args = parser.parse_args()

    height, width = (int(v) for v in args.size.split(","))
    writer = RefWriter(args.out)
    torch.manual_seed(args.seed)

    from ultralytics.nn.tasks import DetectionModel

    if args.weights:
        # Official checkpoint: a pickled DetectionModel. Unpickling pulls in
        # ultralytics classes, so the metadata shim must run before load.
        import importlib.metadata as _md
        _o = _md.version
        _md.version = lambda n: (_o(n) if n != "torchvision" else "0.0.0")
        _ck = torch.load(str(args.weights), map_location="cpu", weights_only=False)
        # AMP-trained checkpoints store half-precision weights; run in float32.
        model = _ck["model"].float().eval()
    else:
        model = DetectionModel(str(CFG), ch=3, nc=args.nc, verbose=False).eval()
        # Random-init BN is ~identity; pull it off identity first so every dumped
        # tensor exercises a non-degenerate BN fold (fold bugs become visible).
        perturb_bns(model)
    detect = model.model[-1]

    captures = {}
    handles = []
    layer_keys = {}
    for index, module in enumerate(model.model):
        if index == len(model.model) - 1:
            continue  # Detect returns a tuple; dumped explicitly below
        key = f"layer{index:02d}_{type(module).__name__}"
        layer_keys[index] = key
        handles.append(module.register_forward_hook(capture_output(captures, key)))
        # P3: per-block isolated checks feed each block its OWN torch input and
        # compare against its output, so the input of every top-level module
        # (except the network INPUT node) must be dumped too.
        handles.append(module.register_forward_hook(capture_input(captures, key + "_in")))

    for index, module in enumerate(model.model):
        if type(module).__name__ == "SPPF":
            prefix = f"layer{index:02d}_sppf"
            handles.append(module.cv1.register_forward_hook(capture_output(captures, prefix + "_y0")))
            handles.append(module.m.register_forward_hook(capture_output(captures, prefix + "_pool")))

    attentions = [m for m in model.modules() if type(m).__name__ == "Attention"]
    for order, attn in enumerate(attentions):
        handles.append(attn.register_forward_hook(capture_input(captures, f"attn{order}_in")))

    for level in range(detect.nl):
        handles.append(detect.one2one_cv2[level].register_forward_hook(capture_output(captures, f"head_box{level}")))
        handles.append(detect.one2one_cv3[level].register_forward_hook(capture_output(captures, f"head_cls{level}")))

    if args.image:
        x = letterbox_image(args.image, height, width)
    else:
        x = torch.randn(1, 3, height, width)
    with torch.no_grad():
        y, preds = model(x)
    for handle in handles:
        handle.remove()

    writer.dump("input", x, "tensor fed to model(x)")

    for index, module in enumerate(model.model[:-1]):
        values = captures[layer_keys[index]]
        assert len(values) == 1, f"{layer_keys[index]} captured {len(values)} times"
        writer.dump(layer_keys[index], values[0], f"output of model.model[{index}] ({type(module).__name__})")
        inkey = layer_keys[index] + "_in"
        if inkey in captures:
            invals = captures[inkey]
            assert len(invals) == 1, f"{inkey} captured {len(invals)} times"
            writer.dump(inkey, invals[0], f"input fed to model.model[{index}] ({type(module).__name__})")

    # SPPF internals + self-check: replaying the module on the dumped input of the
    # previous layer must reproduce the dumped output bit-exactly, otherwise the
    # hooks are attached to the wrong layers.
    for index, module in enumerate(model.model):
        if type(module).__name__ != "SPPF":
            continue
        prefix = f"layer{index:02d}_sppf"
        writer.dump(prefix + "_y0", captures[prefix + "_y0"][0], f"SPPF[{index}].cv1 output")
        stages = captures[prefix + "_pool"]
        assert len(stages) == 3, f"expected 3 chained pools, got {len(stages)}"
        for stage, tensor in enumerate(stages, start=1):
            writer.dump(f"{prefix}_y{stage}", tensor, f"SPPF[{index}] pool stage {stage}")
        with torch.no_grad():
            replayed = module(captures[layer_keys[index - 1]][0])
        err = (replayed - captures[layer_keys[index]][0]).abs().max().item()
        print(f"[check] SPPF replay on layer{index - 1:02d} output: max_abs_err={err:g}")
        assert err == 0.0, "SPPF replay mismatch: hooks attached to the wrong layer"

    folded_by_prefix = export_folds(writer, model, len(model.model) - 1, attentions)

    attn_meta = []
    for order, attn in enumerate(attentions):
        x_attn = captures[f"attn{order}_in"][0]
        with torch.no_grad():
            y_orig = attn.forward(x_attn)
            y_step, inter = attention_stepwise(attn, x_attn)
        err = (y_orig - y_step).abs().max().item()
        print(f"[check] attn{order} stepwise vs torch forward: max_abs_err={err:g}")
        assert err == 0.0, "stepwise dump does not follow the real compute path"
        prefix = f"attn{order}"
        for suffix in ("qkv", "q", "k", "v", "S", "A", "O", "attn_out", "pe_out", "out"):
            writer.dump(f"{prefix}_{suffix}", inter[suffix], f"Attention[{order}] intermediate '{suffix}'")
        # node input + the three conv weight sets (PyTorch layout [c_out,c_in,ky,kx] is
        # identical to FastIV, so the C-side ATTENTION node can load them verbatim).
        writer.dump(f"{prefix}_in", x_attn, f"Attention[{order}] input x")
        # qkv / pe / proj weights are BN-FOLDED (w' = w*s, b' = beta - s*mean):
        # the composite node owns the whole Conv-BN-Identity unit.
        fld = folded_by_prefix[prefix]
        writer.dump(f"{prefix}_qkv_w", fld["qkv"][0], "Attention qkv conv BN-folded weight [qkv_ch,dim,1,1]")
        writer.dump(f"{prefix}_qkv_b", fld["qkv"][1], "Attention qkv conv BN-folded bias [qkv_ch]")
        writer.dump(f"{prefix}_pe_w", fld["pe"][0], "Attention pe dw conv BN-folded weight [dim,1,3,3]")
        writer.dump(f"{prefix}_pe_b", fld["pe"][1], "Attention pe dw conv BN-folded bias [dim]")
        writer.dump(f"{prefix}_proj_w", fld["proj"][0], "Attention proj conv BN-folded weight [dim,dim,1,1]")
        writer.dump(f"{prefix}_proj_b", fld["proj"][1], "Attention proj conv BN-folded bias [dim]")
        attn_meta.append(
            {
                "id": order,
                "dim": attn.dim if hasattr(attn, "dim") else None,
                "num_heads": attn.num_heads,
                "head_dim": attn.head_dim,
                "key_dim": attn.key_dim,
                "scale": attn.scale,
                "pe_groups": attn.pe.conv.groups,
                "in_shape": list(x_attn.shape),
            }
        )

    for level in range(detect.nl):
        writer.dump(f"head_box{level}", captures[f"head_box{level}"][0], f"detect.one2one_cv2[{level}] output")
        writer.dump(f"head_cls{level}", captures[f"head_cls{level}"][0], f"detect.one2one_cv3[{level}] output")

    one2one = preds["one2one"]
    writer.dump("boxes_cat", one2one["boxes"], "concatenated raw box head output [1,4,A]")
    writer.dump("scores_cat", one2one["scores"], "concatenated raw cls head output [1,nc,A]")
    writer.dump("scores_sig", one2one["scores"].sigmoid(), "scores_cat.sigmoid()")
    writer.dump("dbox", detect._get_decode_boxes(one2one), "decoded boxes before stride scaling [1,4,A]")
    writer.dump("anchors", detect.anchors, "make_anchors result [2,A]")
    writer.dump("strides", detect.strides, "per-anchor stride [A]")
    writer.dump("infer_cat", detect._inference(one2one), "concat(dbox*stride, scores.sigmoid()) [1,4+nc,A]")
    writer.dump("detect_out", y, "final postprocessed output [1,k,6]")

    dump_synthetic(writer)

    names = getattr(model, "names", None)
    if names:
        (writer.dir / "names.json").write_text(
            json.dumps({int(k): str(v) for k, v in names.items()}, indent=2) + "\n")
    (writer.dir / "modules.json").write_text(json.dumps(module_table(model, detect), indent=2) + "\n")
    (writer.dir / "attention.json").write_text(json.dumps(attn_meta, indent=2) + "\n")
    count, elems = dump_weights(writer, model)

    lines = []
    for path in sorted(writer.dir.iterdir()):
        if path.name == "MD5SUMS.txt":
            continue
        lines.append(f"{hashlib.md5(path.read_bytes()).hexdigest()}  {path.name}")
    (writer.dir / "MD5SUMS.txt").write_text("\n".join(lines) + "\n")

    print(
        f"wrote {writer.count} tensors, {count} weight blobs ({elems} floats) to {writer.dir}\n"
        f"input=1x3x{height}x{width}  detect_out={tuple(y.shape)}"
    )


if __name__ == "__main__":
    main()
