# import importlib
# import time
# # Core libraries
# import timeit
# import torch
# import torch.nn as nn
# from typing import Optional, Tuple
# import yaml
# from pathlib import Path
# import pandas as pd
# import torch.cuda.nvtx as nvtx


# # Import specific items after reload
# from cs336_basics.annoted_model import BasicsTransformerLM
# from cs336_systems.benchmark import create_random_batch, load_config

# def benchmark_model_with_nvtx(
#     model: nn.Module,
#     batch: torch.Tensor,
#     warmup_steps: int,
#     benchmark_steps: int,
#     forward_only: bool,
#     device: str,
# ) -> Tuple[float, float]:
#     """
#     Benchmark the model's forward and backward passes.
    
#     Args:
#         model: The model to benchmark
#         batch: Input batch tensor
#         warmup_steps: Number of warmup steps
#         benchmark_steps: Number of steps to benchmark
#         forward_only: Whether to only benchmark forward pass
#         device: Device to run on
        
#     Returns:
#         Tuple of (forward_time, backward_time) in seconds
#     """
    
#     with nvtx.range("Warmup"):
#         # Warmup
#         for _ in range(warmup_steps):
#             outputs = model(batch)
#             if not forward_only:
#                 loss = outputs.mean()
#                 loss.backward()
#             if device == "cuda":
#                 torch.cuda.synchronize()
    
#     for step in range(benchmark_steps):
#         with nvtx.range(f"step_{step}"):
#             # Forward pass

#             with nvtx.range("Forward Pass"):
#                 outputs = model(batch)
#                 if device == "cuda":
#                     torch.cuda.synchronize()

#             if not forward_only:
#                 # Backward pass
#                 loss = outputs.mean()

#                 with nvtx.range("Backward Pass"):
#                     loss.backward()
#                     if device == "cuda":
#                         torch.cuda.synchronize()
    

# def main():
#     # Load configuration
#     # time.sleep(10)
#     config_path = Path("configures/benchmark_config.yaml")
#     config = load_config(config_path)
    
#     if torch.cuda.is_available():
#         print("Running on GPU")

#     # Initialize model
#     with nvtx.range("define_model"):
#         model = BasicsTransformerLM(
#             vocab_size=config["vocab_size"],
#             context_length=config["context_length"],
#             d_model=config["d_model"],
#             num_layers=config["num_layers"],
#             num_heads=config["num_heads"],
#             d_ff=config["d_ff"],
#             rope_theta=config["rope_theta"],
#         ).to(config["device"])
    
#     assert next(model.parameters()).is_cuda

#     # Create random batch
#     with nvtx.range("define_input"):
#         batch = create_random_batch(
#             config["batch_size"],
#             config["context_length"],
#             config["vocab_size"],
#             config["device"]
#         )
    
#     # Run benchmark
#     with nvtx.range("nBenchmark"):
#         benchmark_model_with_nvtx(
#             model,
#             batch,
#             config["warmup_steps"],
#             config["benchmark_steps"],
#             config["forward_only"],
#             config["device"]
#         )
    

# if __name__ == "__main__":
#     main() 



import importlib
import time
import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml
from pathlib import Path
import pandas as pd
import torch.cuda.nvtx as nvtx

from cs336_basics.annoted_model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_systems.benchmark import create_random_batch, load_config


def benchmark_model_with_nvtx(
    model: nn.Module,
    batch: torch.Tensor,
    warmup_steps: int,
    benchmark_steps: int,
    forward_only: bool,
    device: str,
) -> Tuple[float, float]:
    """
    Benchmark the model's forward and backward passes including optimizer and cosine LR.
    """
    optimizer = AdamW(model.parameters(), lr=1e-3)
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_iters = warmup_steps
    total_iters = warmup_steps + benchmark_steps
    global_step = 0

    with nvtx.range("Warmup"):
        for _ in range(warmup_steps):
            lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
            for group in optimizer.param_groups:
                group['lr'] = lr

            outputs = model(batch)
            if not forward_only:
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
            global_step += 1

    for step in range(benchmark_steps):
        with nvtx.range(f"step_{step}"):
            with nvtx.range("get_cosine_lr"):
                lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
            for group in optimizer.param_groups:
                group['lr'] = lr

            with nvtx.range("Forward Pass"):
                outputs = model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()

            if not forward_only:
                loss = outputs.mean()
                with nvtx.range("Backward Pass"):
                    loss.backward()
                with nvtx.range("optimizer.step"):
                    optimizer.step()
                    optimizer.zero_grad()
                if device == "cuda":
                    torch.cuda.synchronize()

            global_step += 1

    return 0.0, 0.0  # You can return timing later if needed


def main():
    config_path = Path("configures/benchmark_config.yaml")
    config = load_config(config_path)

    if torch.cuda.is_available():
        print("Running on GPU")

    with nvtx.range("define_model"):
        model = BasicsTransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
        ).to(config["device"])

    assert next(model.parameters()).is_cuda

    with nvtx.range("define_input"):
        batch = create_random_batch(
            config["batch_size"],
            config["context_length"],
            config["vocab_size"],
            config["device"]
        )

    with nvtx.range("nBenchmark"):
        benchmark_model_with_nvtx(
            model,
            batch,
            config["warmup_steps"],
            config["benchmark_steps"],
            config["forward_only"],
            config["device"]
        )


if __name__ == "__main__":
    main()
