#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8
#define TILE_W 8
#define TILE_C 4

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float *__restrict__ x, // input: N x C x H x W
    const float *__restrict__ w, // weights: C_out x C_in x KH x KW
    float *__restrict__ out,     // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w)
{
    extern __shared__ float shmem[]; // shared memory for partial sums

    // TO DO : Tiled matrix multiplication by using shmem

    float *weights = shmem;
    float *inputs = shmem + blockDim.y * TILE_C;

    int M = C_out;
    int L = out_h * out_w;
    int K = C_in * KH * KW;

    int n = blockIdx.z;

    int indexj = blockIdx.x * blockDim.x + threadIdx.x;
    int indexi = blockIdx.y * blockDim.y + threadIdx.y;

    int out_row = indexj / out_w;
    int out_col = indexj % out_w;

    float sum = 0.f;

    for (int c_start = 0; c_start < C_in; c_start += TILE_C)
    {
        for (int kh = 0; kh < KH; kh++)
        {
            for (int kw = 0; kw < KW; kw++)
            {
                for (int t = threadIdx.x; t < TILE_C; t += blockDim.x)
                {
                    int c = c_start + t;
                    float value = 0.0f;

                    if (indexi < M && c < C_in)
                    {
                        int weight_index = ((indexi * C_in + c) * KH + kh) * KW + kw;
                        value = w[weight_index];
                    }

                    weights[threadIdx.y * TILE_C + t] = value;
                }

                int row = out_row * stride + kh - pad;
                int col = out_col * stride + kw - pad;

                for (int t = threadIdx.y; t < TILE_C; t += blockDim.y)
                {
                    int c = c_start + t;
                    float value = 0.0f;

                    if (indexj < L && c < C_in && row >= 0 && row < H && col >= 0 and col < W)
                    {
                        int input_index = ((n * C_in + c) * H + row) * W + col;
                        value = x[input_index];
                    }
                    inputs[t * blockDim.x + threadIdx.x] = value;
                }

                __syncthreads();

                for (int t = 0; t < TILE_C; t++)
                {
                    sum += weights[threadIdx.y * TILE_C + t] * inputs[t * blockDim.x + threadIdx.x];
                }

                __syncthreads();
            }
        }
    }
    if (indexi < M && indexj < L)
    {
        out[(n * M + indexi) * L + indexj] = sum;
    }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                        int stride, int pad)
{
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = 1 + (H + 2 * pad - KH) / stride;
    int out_w = 1 + (W + 2 * pad - KW) / stride;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(TILE_H, TILE_W);
    dim3 grid((out_w * out_h + block.x - 1) / block.x,
              (C_out + block.y - 1) / block.y,
              N);
    size_t shmem_bytes = (TILE_C * (TILE_H + TILE_W)) * sizeof(float);

    gemm_gpu_o4_kernel<<<grid, block, shmem_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
