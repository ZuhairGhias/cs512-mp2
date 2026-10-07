import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import time
import matplotlib.pyplot as plt
from pathlib import Path
image_path = Path("images")
image_path.mkdir(exist_ok=True)

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 

        # Precompute output size
        self.out_h = 1 + (H + 2*padding - kernel_size) // stride
        self.out_w = 1 + (W + 2*padding - kernel_size) // stride

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.

        patches = []

        for kh in range(KH):
            for kw in range(KW):
                patch = x_pad[
                    :, # the whole batch
                    :, # all colors
                    kh:kh + out_h * S:S,
                    kw:kw + out_w * S:S,
                ]

                patches.append(patch)

        patches = torch.stack(patches, dim=-1)
        return patches.permute(0, 2, 3, 1, 4).reshape(N, out_h * out_w, C * KH * KW)

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        C = self.in_channels
        KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)          

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        w = self.weight.reshape(C_out, C*KH*KW)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        out = torch.matmul(cols, w.T)

        # TO DO: 4) Add bias.
        out += self.bias

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out =  out.transpose(1, 2).reshape(N, C_out, self.out_h, self.out_w)

        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    N, C, H, W = 2, 4, 22, 22
    x = torch.randn(N, C, H, W)
    out_channels=8
    kernel_size=7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1)
    out = model(x)

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))

    N, C, H, W = 2, 4, 64, 64
    out_channels=8
    kernel_size=7
    image_times = []
    image_sizes = [22, 44, 66, 88]
    for image_size in image_sizes:
        x = torch.randn(N, C, image_size, image_size)
        model = ConvModel(image_size, image_size, C, out_channels, kernel_size, stride=1, padding=1)

        # warmup
        model(x)
        torch.cuda.synchronize() #https://discuss.pytorch.org/t/why-we-use-torch-cuda-synchronize/193111/2

        times = []
        for _ in range(5):
            start = time.perf_counter()
            out = model(x)
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
    
            # warmup
            model(x)
            torch.cuda.synchronize() #https://discuss.pytorch.org/t/why-we-use-torch-cuda-synchronize/193111/2
            
            times = []
            for _ in range(5):
                start = time.perf_counter()
                out = model(x)
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

    fig.suptitle("Eager PyTorch")
    fig.tight_layout()
    fig.savefig(image_path / "pytorch_timings.png")

    # with profile(activities=[ProfilerActivity.CUDA]) as prof:
    #     with record_function("manual_convolution"):
    #         out = model(x)

    # prof.export_chrome_trace("trace.json")
        
