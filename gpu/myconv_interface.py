import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.cpp_extension import load

# Compile and load CUDA extension
conv_module = load(name="myconv",
                     sources=[str(Path(__file__).with_name("myconv_kernel.cu"))],
                     verbose=True)

# Input parameters
N, C_in, H, W = 4, 3, 25, 25
C_out, KH, KW = 4, 6, 6
stride, pad = 1, 1

# Allocate tensors
x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
w = torch.randn(C_out, C_in, KH, KW, device="cuda", dtype=torch.float32)

# Run o4 kernel
out_custom = conv_module.conv_cuda(x, w, stride, pad)

# Reference solution (PyTorch)
out_ref = torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)

# Test shape and correctness
print("CUDA --- shape check:", out_custom.shape == out_ref.shape)
print("CUDA --- correctness check:", torch.allclose(out_custom, out_ref, atol=1e-4))

image_path = Path("images")
image_path.mkdir(exist_ok=True)
torch.manual_seed(0)
N, C, H, W = 2, 4, 64, 64
out_channels = 8
kernel_size = 7
stride, pad = 1, 1

image_times = []
image_sizes = [22, 44, 66, 88]
for image_size in image_sizes:
    x = torch.randn(N, C, image_size, image_size, device="cuda")
    w = torch.randn(out_channels, C, kernel_size, kernel_size, device="cuda")
    bias = torch.zeros(1, out_channels, 1, 1, device="cuda")

    # Warmup; include bias addition as in the other implementations.
    out = conv_module.conv_cuda(x, w, stride, pad) + bias
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = conv_module.conv_cuda(x, w, stride, pad) + bias
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    image_times.append(avg)
    print(f"image size - {image_size}: {avg:.3f} ms")

kernel_times = []
K_sizes = [3, 5, 7, 11]
for K_size in K_sizes:
    x = torch.randn(N, C, H, W, device="cuda")
    w = torch.randn(out_channels, C, K_size, K_size, device="cuda")
    bias = torch.zeros(1, out_channels, 1, 1, device="cuda")

    # Warmup
    out = conv_module.conv_cuda(x, w, stride, pad) + bias
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = conv_module.conv_cuda(x, w, stride, pad) + bias
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    kernel_times.append(avg)
    print(f"kernel size - {K_size}: {avg:.3f} ms")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, sizes, times, parameter in [
    (axes[0], image_sizes, image_times, "Image"),
    (axes[1], K_sizes, kernel_times, "Kernel"),
]:
    ax.plot(sizes, times, marker="o")
    ax.set_title(f"Varying {parameter.lower()} size")
    ax.set_xlabel(f"{parameter} side length (pixels)")
    ax.set_ylabel("Mean wall time per call (ms)")
    ax.set_xticks(sizes)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

fig.suptitle("Custom CUDA")
fig.tight_layout()
fig.savefig(image_path / "cuda_timings.png")
