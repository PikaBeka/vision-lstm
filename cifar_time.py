from vision_lstm.vision_lstm2 import VisionLSTM2
from vision_lstm.vision_minlstm import VisionMinLSTM
import torch
import time
from contextlib import nullcontext
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducible-ish perf knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x
torch.backends.cudnn.benchmark = True       # fixed shape speedup

# ----- build your model here -----
model = VisionLSTM2(
    dim=384, input_shape=(3, 224, 224), patch_size=16, depth=12,
    pooling='bilateral_avg', output_shape=(1000,)
).to(device).eval()

# model = VisionMinLSTM(
#     dim=384, input_shape=(3, 224, 224), patch_size=16, depth=12,
#     pooling='bilateral_avg', output_shape=(1000,)
# ).to(device).eval()

flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224, device=device).to(
    memory_format=torch.channels_last))
print("Total FLOPs:", flops.total())

print(parameter_count_table(model))

# choose dtype + autocast for realistic inference
amp_ctx = torch.autocast(
    device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def bench(batch, iters_warmup=50, iters_timed=300):
    x = torch.randn(batch, 3, 224, 224, device=device).to(
        memory_format=torch.channels_last)
    # warmup
    with torch.inference_mode(), amp_ctx:
        for _ in range(iters_warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # timing with CUDA events
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    times_ms = []
    with torch.inference_mode(), amp_ctx:
        for _ in range(iters_timed):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))  # ms

    import numpy as np
    times = np.array(times_ms)
    avg = times.mean()
    p50, p90, p99 = np.percentile(times, [50, 90, 99])
    thr = (batch * 1000.0) / avg
    return dict(batch=batch, avg_ms=avg, p50_ms=p50, p90_ms=p90, p99_ms=p99, imgs_per_s=thr)


# run a few meaningful batches
results = [bench(b) for b in [1, 2, 4, 8, 16, 32]]
for r in results:
    print(f"B={r['batch']:>2} | avg {r['avg_ms']:.2f} ms | p50 {r['p50_ms']:.2f} | p90 {r['p90_ms']:.2f} | p99 {r['p99_ms']:.2f} | {r['imgs_per_s']:.1f} img/s")
