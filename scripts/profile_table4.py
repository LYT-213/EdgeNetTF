#!/usr/bin/env python3
"""Unified computational profiling for manuscript Table 4.

Run from the repository root in a Kaggle notebook or equivalent environment.
Latency includes model forward inference only; data loading, window generation,
preprocessing, and FFT generation are excluded.
"""

from pathlib import Path
import io
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgenettf.models import EdgeNetTF
from edgenettf.baselines import (
    CNN1D,
    MobileNetV2_1D,
    SqueezeNet1D,
    ShuffleNetV2_1D,
    ResNet1D,
    DenseNet1D,
    InceptionTime1D,
)

try:
    from thop import profile
except ImportError as exc:
    raise ImportError("Install THOP first: pip install thop") from exc

NUM_CHANNELS = 12
TIME_LENGTH = 600
FREQ_LENGTH = 300
CPU_THREADS = 1
WARMUP = 100
REPEAT = 1000
OUTPUT_CSV = ROOT / "results" / "table4_computational_efficiency_rerun.csv"
OUTPUT_ENV = ROOT / "results" / "table4_environment_rerun.txt"

torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def build_models():
    return {
        "CNN1D": CNN1D(),
        "MobileNetV2_1D": MobileNetV2_1D(),
        "SqueezeNet1D": SqueezeNet1D(),
        "ShuffleNetV2_1D": ShuffleNetV2_1D(),
        "ResNet1D": ResNet1D(),
        "DenseNet1D": DenseNet1D(),
        "InceptionTime1D": InceptionTime1D(),
        "EdgeNetTF": EdgeNetTF(),
    }


def make_inputs(name):
    torch.manual_seed(42)
    x_time = torch.randn(1, NUM_CHANNELS, TIME_LENGTH)
    if name == "EdgeNetTF":
        x_freq = torch.randn(1, NUM_CHANNELS, FREQ_LENGTH)
        return (x_time, x_freq)
    return (x_time,)


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def macs_and_flops(model, inputs):
    model = model.cpu().eval()
    inputs = tuple(x.cpu() for x in inputs)
    macs, _ = profile(model, inputs=inputs, verbose=False)
    return float(macs), float(2.0 * macs)


def cpu_latency(model, inputs):
    model = model.cpu().eval()
    inputs = tuple(x.cpu() for x in inputs)
    times = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(*inputs)
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            model(*inputs)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return arr.mean(), arr.std(ddof=0)


def gpu_latency(model, inputs):
    if not torch.cuda.is_available():
        return np.nan, np.nan
    model = model.cuda().eval()
    inputs = tuple(x.cuda() for x in inputs)
    times = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(*inputs)
        torch.cuda.synchronize()
        for _ in range(REPEAT):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    arr = np.asarray(times)
    return arr.mean(), arr.std(ddof=0)


def write_environment():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    lines = [
        f"Python: {platform.python_version()}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA runtime reported by PyTorch: {torch.version.cuda}",
        f"GPU: {gpu}",
        f"CPU: {platform.processor()}",
        f"CPU threads used for profiling: {CPU_THREADS}",
        f"Warm-up iterations: {WARMUP}",
        f"Timed iterations: {REPEAT}",
        "Batch size: 1",
        f"Temporal input shape: (1, {NUM_CHANNELS}, {TIME_LENGTH})",
        f"Frequency input shape for EdgeNetTF: (1, {NUM_CHANNELS}, {FREQ_LENGTH})",
        "Latency scope: model forward pass only",
        "Excluded from latency: data loading, window generation, preprocessing, FFT",
        "FLOPs convention: 2 x MACs",
    ]
    OUTPUT_ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    write_environment()
    rows = []
    for name in build_models():
        model = build_models()[name]
        inputs = make_inputs(name)
        params = trainable_params(model)
        size_mb = state_dict_size_mb(model)
        macs, flops = macs_and_flops(model, inputs)

        cpu_model = build_models()[name]
        gpu_model = build_models()[name]
        cpu_mean, cpu_sd = cpu_latency(cpu_model, inputs)
        gpu_mean, gpu_sd = gpu_latency(gpu_model, inputs)

        row = {
            "Model": name,
            "Params_M": params / 1e6,
            "Model_Size_MB": size_mb,
            "MACs_M": macs / 1e6,
            "FLOPs_M": flops / 1e6,
            "CPU_Latency_Mean_ms": cpu_mean,
            "CPU_Latency_SD_ms": cpu_sd,
            "GPU_Latency_Mean_ms": gpu_mean,
            "GPU_Latency_SD_ms": gpu_sd,
        }
        rows.append(row)
        print(
            f"{name}: {row['Params_M']:.4f}M params | "
            f"{row['MACs_M']:.2f}M MACs | {row['FLOPs_M']:.2f}M FLOPs | "
            f"CPU {cpu_mean:.3f}±{cpu_sd:.3f} ms | "
            f"GPU {gpu_mean:.3f}±{gpu_sd:.3f} ms"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_ENV}")


if __name__ == "__main__":
    main()
