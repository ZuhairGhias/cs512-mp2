import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    TILE = nl.tile_size.pmax

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for out_c in nl.affine_range(out_channels // TILE):
            out_start = out_c * TILE
            out_end = out_start + TILE

            bias_tile = nl.load(bias[out_start:out_end]).reshape((TILE,1))

            for h in nl.affine_range(out_height):
                sum = nl.zeros((TILE, out_width), dtype=nl.float32, buffer=nl.psum,)

                for in_c in nl.affine_range(in_channels // TILE):
                    in_start = in_c * TILE
                    in_end = in_start + TILE
                    weight_block = nl.load(W[out_start:out_end, in_start:in_end, :, :])

                    #apply the kernel
                    for kh in nl.affine_range(filter_height):
                        for kw in nl.affine_range(filter_width):
                            weights = nl.copy(weight_block[:, :, kh, kw])
                            inputs = nl.load(X[b,in_start:in_end,h+kh,kw:kw+out_width])

                            sum += nl.matmul(weights,inputs)

                result = nl.copy(sum, dtype=nl.float32)
                result = nl.add(result, bias_tile)
                result = nl.copy(result, dtype=X.dtype)
                nl.store(X_out[b, out_start:out_end, h, :], value=result)
        
    return X_out

