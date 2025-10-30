import torch
import time
import re
from pathlib import Path
from tqdm import tqdm

# --- import your model class ---
from vision_lstm.vision_minlstm import VisionMinLSTM
from vision_lstm.vision_lstm2 import VisionLSTM2

# ---------- utilities ----------


def load_state_dict_flex(path, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location)
    # Common layouts: pure state_dict OR dict with 'state_dict'/'model'
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # If it looks like a pure state_dict already:
        if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
            return ckpt
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def infer_visminlstm_config(sd):
    """
    Infer: dim, patch_size, seqlens(H',W'), input resolution, depth, num_classes, pooling head size
    """
    # dim from embed sizes (pos_embed or patch_embed)
    # pos embed: 'pos_embed.embed' -> shape (1, H', W', dim)
    if "pos_embed.embed" in sd:
        _, Hs, Ws, dim = sd["pos_embed.embed"].shape
        seqlens = (Hs, Ws)
    else:
        # fallback to patch_embed.proj.weight: [dim, C, ph, pw]
        w = sd["patch_embed.proj.weight"]           # (dim, C, ph, pw)
        dim = w.shape[0]
        ph, pw = w.shape[-2], w.shape[-1]
        # seqlens via classification head or layers is trickier; leave None
        seqlens = None

    # patch size from patch_embed.proj.weight
    ph, pw = sd["patch_embed.proj.weight"].shape[-2:]
    patch_size = ph if ph == pw else (ph, pw)

    # Input resolution = seqlens * patch_size when seqlens known
    if seqlens is not None:
        if isinstance(patch_size, int):
            H = seqlens[0] * patch_size
            W = seqlens[1] * patch_size
        else:
            H = seqlens[0] * patch_size[0]
            W = seqlens[1] * patch_size[1]
        input_shape = (3, H, W)
    else:
        # Fall back to common cases
        input_shape = (3, 224, 224)

    # depth from layers.* blocks (ModuleList index)
    layer_idxs = []
    p = re.compile(r"^layers\.(\d+)\.")
    for k in sd.keys():
        m = p.match(k)
        if m:
            layer_idxs.append(int(m.group(1)))
    depth = (max(layer_idxs) + 1) if layer_idxs else 12

    # classes & head dim
    num_classes = 1000
    head_dim = dim
    if "head.weight" in sd:
        num_classes = sd["head.weight"].shape[0]
        head_dim = sd["head.weight"].shape[1]

    # Pooling: in VisionMinLSTM, "bilateral_flatten" -> head_dim == 2 * dim
    pooling = "bilateral_flatten" if head_dim == 2 * \
        dim else "bilateral_avg" if head_dim == dim else "bilateral_flatten"

    return {
        "dim": dim,
        "patch_size": patch_size,
        "input_shape": input_shape,
        "depth": depth,
        "num_classes": num_classes,
        "pooling": pooling,
    }


def build_model_from_sd(sd):
    cfg = infer_visminlstm_config(sd)
    model = VisionLSTM2(
        dim=cfg["dim"],
        input_shape=cfg["input_shape"],
        patch_size=cfg["patch_size"],
        depth=cfg["depth"],
        output_shape=(cfg["num_classes"],),
        pooling=cfg["pooling"],
        # keep other defaults; they match the implementation
    )
    # Strict load first; if that fails due to minor naming diffs, relax
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print("[warn] strict load failed, retrying with strict=False\n", e)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print("  missing:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("  unexpected:", unexpected[:8], "..." if len(
                unexpected) > 8 else "")
    return model, cfg


# ---------- pick your checkpoint ----------
# Prefer a *model* checkpoint, e.g. 'vislstm cp=latest model.th'
ckpt_path = Path("checkpoints/3i471hv1/sx0hjr39/checkpoints") / \
    "vislstm cp=latest model.th"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device:", device)

# ---------- build & load ----------
sd = load_state_dict_flex(ckpt_path, map_location=device)
model, cfg = build_model_from_sd(sd)
model.to(device).eval()
print("Loaded VisionMinLSTM with:", cfg)

# ---------- dummy input that *matches* the trained resolution ----------
B = 32
_, C, H, W = (B, *cfg["input_shape"])
x = torch.randn(B, C, H, W, device=device)

# ---------- benchmarking (warmup + synced timing) ----------
torch.backends.cudnn.benchmark = True  # speed up convs on fixed shapes
iters_warmup = 50
iters_timed = 500

with torch.inference_mode():
    for _ in range(iters_warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_ms = 0.0
    for _ in tqdm(range(iters_timed)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms += (time.time() - t0) * 1000.0

avg_ms = total_ms / iters_timed
throughput = (B * 1000.0) / avg_ms
print(
    f"Avg latency: {avg_ms:.3f} ms  |  Throughput: {throughput:.1f} img/s @ batch={B}")
