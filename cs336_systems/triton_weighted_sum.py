import triton
import torch
import triton.language as tl
from triton.runtime import driver
from einops import rearrange
import math

# Define for Forward Pass for normal approach
def weighted_sum(x, weight):
    # Assume x has n-dim shape, [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)

@triton.jit
def weighted_sum_fwd(
    x_ptr, 
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    # Every instance will compute the weighted sum of atile of rows of x
    # 'tl.program_id' gives us a way to get the starting pointer address
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the momory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory form major to minor
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100.

    x_block_ptr = tl.make_block_ptr(
        x_ptr, 
        shape = (ROWS, D, ),
        strides = (x_stride_row, x_stride_dim),
        offsets = (row_tile_idx * ROWS_TILE_SIZE, 0), # Starts from row row_tile_idx * ROWS_TILE_SIZE, column 0
        block_shape = (ROWS_TILE_SIZE, D_TILE_SIZE),
        order = (1, 0),
    )

    # -----------------------------------------------------------------------------
    # tl.make_block_ptr(...)
    # -----------------------------------------------------------------------------
    # Creates a block pointer that represents a rectangular (multi-dimensional) 
    # region of memory, which can be used for efficient vectorized memory access.
    #
    # This is especially useful for tiled workloads, such as matrix multiplication, 
    # where each program instance processes a tile (sub-block) of the full tensor.
    #
    # Parameters:
    # - ptr: the base pointer to the beginning of the tensor (e.g., x_ptr).
    # - shape: the full shape of the original tensor (e.g., (ROWS, COLS)).
    # - strides: the strides (step sizes in memory) for each dimension.
    # - offsets: the starting index in each dimension for the block.
    # - block_shape: the shape of the block to be processed (e.g., (BLOCK_ROWS, BLOCK_COLS)).
    # - order: a permutation of dimensions that defines the memory access order
    #          from major to minor (e.g., order=(1, 0) means column-major first).
    #
    # The resulting block pointer supports:
    # - Efficient `tl.load()` and `tl.store()` over blocks
    # - Fine-grained control over memory tiling, coalescing, and boundary handling
    # - Pointer arithmetic via `.advance()` to move to the next tile in loops
    #
    # Example:
    # x_block_ptr = tl.make_block_ptr(
    #     x_ptr,
    #     shape=(ROWS, D),
    #     strides=(x_stride_row, x_stride_dim),
    #     offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    #     block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    #     order=(1, 0)
    # )
    # -----------------------------------------------------------------------------

    # `order=(1, 0)` means that dimension 1 (columns) is traversed first,
    # and dimension 0 (rows) is traversed second — i.e., row-major order.
    # This typically ensures better memory coalescing on CUDA devices.

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape = (D, ),
        strides = (weight_stride_dim, ),
        offsets = (0, ),
        block_shape = (D_TILE_SIZE, ),
        order = (0, ),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape = (ROWS,),
        strides = (output_stride_row, ),
        offsets = (row_tile_idx * ROWS_TILE_SIZE,),
        block_shape = (ROWS_TILE_SIZE, ),
        order = (0, )
    )

    output = tl.zeros((ROWS_TILE_SIZE, ), dtype=tl.float32)

    # We are dealing within 1 tile
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check = (0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check = (0, ), padding_option="zero")

        # Compute the weighted sum of the row
        output += tl.sum(row * weight[None, :], axis = 1)

        # Move the pointers to the next tile
        # These are (rows, columns) coordinates detlas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # Move by D_TILE_SIZE in columns
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))

    # Write output to the output block pointer (a single scalar per row)
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check = (0, ))

@triton.jit
def weighted_sum_backward(
        x_ptr, weight_ptr,
        grad_output_ptr,
        grad_x_ptr, partial_grad_weight_ptr,
        stride_xr, stride_xd,
        stride_wd,
        stride_gr,
        stride_gxr, stride_gxd,
        stride_gwb, stride_gwd,
        NUM_ROWS, D, 
        ROWS_TILE_SIZE: tl.constexpr,
        D_TILE_SIZE: tl.constexpr,
    ):
        row_tile_idx = tl.program_id(0)
        n_row_tiles = tl.num_programs(0)
        
        # inputs
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(NUM_ROWS, ), strides = (stride_gr,),
            offsets = (row_tile_idx * ROWS_TILE_SIZE, ),
            block_shape = (ROWS_TILE_SIZE, ),
            order=(0,)

        )

        x_block_ptr = tl.make_block_ptr(
            x_ptr,
            shape=(NUM_ROWS, D, ), strides=(stride_xr, stride_xd),
            offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
            block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0)
        )

        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(D, ), strides=(stride_wd,),
            offsets=(0, ), block_shape=(D_TILE_SIZE, ),
            order=(0,)
        )


        grad_x_block_ptr = tl.make_block_ptr(
            grad_x_ptr,
            shape = (NUM_ROWS, D, ), strides=(stride_gxr, stride_gxd),
            offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), 
            block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )

        partial_grad_weight_block_ptr = tl.make_block_ptr(
            partial_grad_weight_ptr, 
            shape=(n_row_tiles, D, ), strides=(stride_gwb, stride_gwd),
            offsets=(row_tile_idx, 0),
            block_shape=(1, D_TILE_SIZE),
            order=(1,0)
        )

        for i in range(tl.cdiv(D, D_TILE_SIZE)):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, ), padding_option="zero") # (ROWS_TILE_SIZE,)

            # Outer product for grad_x
            weight = tl.load(weight_block_ptr, boundary_check=(0, ), padding_option="zero") # (D_TILE_SIZE, )
            grad_x_row = grad_output[:, None] * weight[None, :]
            tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0,1))

            # Reduce as many rows as possible for the grad_weight result
            row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_weight_row = tl.sum(row * grad_output[:None], axis = 0, keep_dim = True)
            tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # Never out of bounds for dim 0
            
            # Move the pointers to the next tile along D
            x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
            weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))
            partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
            grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in backward pass
        # when we only receive the gradient, with respect tothe output tensor
        # and we need to compute the gradients w.r.t. x and weight

        D, output_dims = x.shape[-1], x.shape[:-1] # Support high dimensions

        # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Weight must be 1D and have the same length as the last dimension of x"
        assert x.is_cuda and weight.is_cuda, "x and weight must be on Nvidia GPU"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) # Get the ceiling power of 2 for D
        ctx.ROWS_TILE_SIZE = 16 # Manually set the tile size for rows

        y = torch.empty(output_dims, device=x.device)
        
        # Lanuch kerneal with warm up
        n_rows = y.numel()
        weighted_sum_fwd[(math.ceil(n_rows / ctx.ROWS_TILE_SIZE), )](
            x, weight,
            y,
            x.stride(0), x.stride(1), # Memory step size between rows and columns
            weight.stride(0),
            y.stride(0),
            ROWS = n_rows, D = D,
            ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE = ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1]) #Reshape the tensor y to match the shape of input_shape excluding its last dimension.
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # Our strategy is for each thread block to first write to a partial buffer
        # then we receive over this buffer to get the final gradient

        partial_grad_weight = torch.empty((tl.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(math.ceil(n_rows / ctx.ROWS_TILE_SIZE), )](
             x, weight,
             grad_out,
             grad_x, partial_grad_weight,
             x.stride(0), x.stride(1),
             weight.stride(0),
             grad_out.stride(0),
             grad_x.stride(0), grad_x.stride(1),
             partial_grad_weight.stride(0), partial_grad_weight.stride(1),
             NUM_ROWS=n_rows, D=D,
             ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE
        )


        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight

def main():
    # Set device
    device = torch.device("cuda")

    # Create input tensor x of shape (batch_size, D)
    batch_size = 10240
    D = 7000
    x = torch.randn(batch_size, D, device=device, dtype=torch.float32, requires_grad=True)

    # Create 1D weight tensor of shape (D,)
    weight = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

    # Define function
    f_weightedsum = WeightedSumFunc.apply

    # Compute output
    y = f_weightedsum(x, weight)


    # Compute expected output using PyTorch
    expected_y = (x * weight).sum(dim=-1)

    # Assert that the outputs are close within tolerance
    assert torch.allclose(y, expected_y, rtol=1e-3, atol=1e-4), (
        "Mismatch between custom and torch implementation",
        y,
        expected_y
    )

    # Print result
    print("Output:")
    print(y)
    print(f"Shape: {y.shape}")
    print(f"Device: {y.device}")
    print(f"Requires Grad: {y.requires_grad}")
    print(f"Grad Fn: {y.grad_fn}")

if __name__ == "__main__":
    main()