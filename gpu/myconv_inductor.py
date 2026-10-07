import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import time
import matplotlib.pyplot as plt
from pathlib import Path
image_path = Path("images")
image_path.mkdir(exist_ok=True)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    # Instantiate your PyTorch model
    N, C, H, W = 2, 3, 19, 19
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    scripted_model = torch.compile(model, backend="inductor")
    out = scripted_model(x)
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))

    N, C, H, W = 2, 4, 64, 64
    out_channels=8
    kernel_size=7

    image_times = []
    image_sizes = [22, 44, 66, 88]
    for image_size in image_sizes:
        x = torch.randn(N, C, image_size, image_size)
        model = ConvModel(image_size, image_size, C, out_channels, kernel_size, stride=1, padding=1)
        torch.compiler.reset()
        scripted_model = torch.compile(model, backend="inductor")

        # warmup
        scripted_model(x)
        torch.cuda.synchronize() #https://discuss.pytorch.org/t/why-we-use-torch-cuda-synchronize/193111/2

        times = []
        for _ in range(5):
            start = time.perf_counter()
            out = scripted_model(x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        image_times.append(avg)
        print(f"image size - {image_size}: {avg:.3f} ms")

    kernel_times = []
    K_sizes = [3, 5, 7, 11]
    for K_size in K_sizes:
            x = torch.randn(N, C, H, W)
            model = ConvModel(H, W, C, out_channels, K_size, stride=1, padding=1)
            torch.compiler.reset()
            scripted_model = torch.compile(model, backend="inductor")

            # warmup
            scripted_model(x)
            torch.cuda.synchronize() #https://discuss.pytorch.org/t/why-we-use-torch-cuda-synchronize/193111/2
            
            times = []
            for _ in range(5):
                start = time.perf_counter()
                out = scripted_model(x)
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

    fig.suptitle("Inductor PyTorch")
    fig.tight_layout()
    fig.savefig(image_path / "pytorch_timings_inductor.png")