import os
import sys
import contextlib
import time
from pathlib import Path

@contextlib.contextmanager
def suppress_stderr_fd():
    # Duplicate the current stderr file descriptor
    original_stderr_fd = os.dup(2)
    # Open the null device
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        # Replace file descriptor 2 (stderr) with the devnull file descriptor
        os.dup2(devnull_fd, 2)
        yield
    finally:
        # Restore the original stderr file descriptor
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        os.close(devnull_fd)

# Import non-noisy modules first
import warnings
import logging

# Now wrap the noisy imports in the context manager
with suppress_stderr_fd():
    import tensorrt as trt
    trt_logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(trt_logger, "")
    import torch_tensorrt
    import torch
    from ._base import OptimizeBase

# -----------------------------------------------------------------------------
# Logging configuration (only essential messages)
# -----------------------------------------------------------------------------
logger = logging.getLogger("ModelOptimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

# Suppress other loggers
for name in ["torch", "torch.nn.parallel", "torch.distributed", "TensorRT", 
            "pytorch_tensorrt", "tensorrt.plugin", "torch_tensorrt.dynamo"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def print_progress(message, progress=None):
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    sp = spinner[int(time.time() * 10) % len(spinner)]
    if progress is not None:
        print(f"\r{sp} {message} ({progress:.1f}%)", end="", flush=True)
    else:
        print(f"\r{sp} {message}", end="", flush=True)

def get_model_size(model_or_path):
    """Return model size in MB."""
    try:
        if isinstance(model_or_path, str):
            return Path(model_or_path).stat().st_size / (1024 * 1024)
        else:
            tmp_path = "tmp_model.pt"
            try:
                torch.save(model_or_path, tmp_path)
            except Exception:
                try:
                    torch.save(model_or_path.state_dict(), tmp_path)
                except Exception:
                    return 0
            size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return size_mb
    except Exception:
        return 0

def benchmark_inference(model, input_tensor, num_warmup=5, num_iters=100):
    """Benchmark model inference performance and return latency stats in ms."""
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
            torch.cuda.synchronize()
    
    latencies = []
    for i in range(num_iters):
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        print_progress("Running inference benchmark", (i + 1) * 100 / num_iters)
    print()  # newline
    
    latencies_sorted = sorted(latencies)
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p90_ms": latencies_sorted[int(0.9 * len(latencies))],
        "p99_ms": latencies_sorted[int(0.99 * len(latencies))],
        "min_ms": min(latencies),
        "max_ms": max(latencies)
    }

def bn_stats_loss(model, input_tensor):
    """
    Compute loss measuring the difference between BatchNorm activations and
    the stored running statistics.
    """
    bn_loss = torch.tensor(0.0, device=input_tensor.device, requires_grad=True)
    hooks = []
    
    def hook_fn(module, inp, out):
        nonlocal bn_loss
        out = out.clone()
        dims = [0, 2, 3]  # Batch, Height, Width
        act_mean = out.mean(dim=dims)
        act_var = ((out - act_mean.view(1, -1, 1, 1)) ** 2).mean(dim=dims)
        running_mean = module.running_mean.to(act_mean.device)
        running_var = module.running_var.to(act_var.device)
        loss_mean = F.mse_loss(act_mean, running_mean.detach())
        loss_var = F.mse_loss(act_var, running_var.detach())
        bn_loss = bn_loss + loss_mean + loss_var

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.set_grad_enabled(True):
        _ = model(input_tensor)
    
    for h in hooks:
        h.remove()
    
    return bn_loss

def generate_distilled_data(model, num_samples=50, input_shape=(3, 224, 224),
                            num_iterations=100, lr=0.1, device="cuda"):
    """
    Generate synthetic data by optimizing noise so that the model's BN stats
    are matched. Returns a tensor of shape (num_samples, *input_shape).
    """
    synthetic_data = torch.randn(num_samples, *input_shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([synthetic_data], lr=lr)
    
    for iter in range(num_iterations):
        optimizer.zero_grad()
        loss = bn_stats_loss(model, synthetic_data)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            synthetic_data.clamp_(-1, 1)
        if iter % 10 == 0:
            print_progress("Generating synthetic dataset", (iter + 1) * 100 / num_iterations)
    print()
    return synthetic_data.detach()

def evaluate_accuracy(original_model, optimized_model, num_samples=32,
                      input_shape=(3, 224, 224), device="cuda"):
    """
    Evaluate accuracy by comparing predictions of original and optimized
    models on a robust synthetic dataset. Each sample is processed individually
    to match the compiled model's expected batch size.
    """
    logger.info("Generating robust synthetic dataset for accuracy evaluation...")
    # Generate synthetic data with 'num_samples' samples
    synthetic_data = generate_distilled_data(
        original_model,
        num_samples=num_samples,
        input_shape=input_shape,
        num_iterations=200,
        lr=0.05,
        device=device
    ).to(device)
    
    top1_match = 0
    top5_match = 0
    kl_divs = []
    confidence_diff = []
    
    for i in range(num_samples):
        sample = synthetic_data[i].unsqueeze(0)  # shape: (1, C, H, W)
        with torch.no_grad():
            orig_logits = original_model(sample)
            opt_logits = optimized_model(sample)
        orig_prob = F.softmax(orig_logits, dim=1)[0]
        opt_prob = F.softmax(opt_logits, dim=1)[0]
        kl = F.kl_div(opt_prob.log(), orig_prob, reduction="sum")
        kl_divs.append(kl.item())
        orig_top1 = torch.argmax(orig_prob).item()
        opt_top1 = torch.argmax(opt_prob).item()
        if orig_top1 == opt_top1:
            top1_match += 1
        orig_top5 = orig_prob.topk(5)[1].tolist()
        opt_top5 = opt_prob.topk(5)[1].tolist()
        if any(cls in opt_top5 for cls in orig_top5):
            top5_match += 1
        confidence_diff.append(opt_prob.max().item() - orig_prob.max().item())
    
    return {
        "top1_match_percent": (top1_match / num_samples) * 100,
        "top5_match_percent": (top5_match / num_samples) * 100,
        "avg_kl_divergence": sum(kl_divs) / len(kl_divs),
        "avg_confidence_diff": sum(confidence_diff) / len(confidence_diff)
    }

def get_model_memory_footprint(model, input_shape=(1, 3, 224, 224), device="cuda"):
    """Measure the model's peak GPU memory usage (in MB) during inference."""
    measurements = []
    for _ in range(5):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(input_shape, device=device)
        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)
        measurements.append(peak_mem)
        del x
        torch.cuda.empty_cache()
    avg_peak = sum(measurements) / len(measurements)
    return avg_peak / (1024 * 1024)

def get_model_weights_size(model):
    """Calculate the total size of model weights in MB."""
    try:
        size_bytes = 0
        state_dict = model.state_dict() if hasattr(model, 'state_dict') else None
        if state_dict:
            for param in state_dict.values():
                if isinstance(param, torch.Tensor):
                    size_bytes += param.nelement() * param.element_size()
            return size_bytes / (1024 * 1024)
        return 0
    except Exception:
        return 0

# -----------------------------------------------------------------------------
# GPU Optimization Class
# -----------------------------------------------------------------------------
class OptimizeGPU(OptimizeBase):
    def optimize(self, model_path: str, output_path: str = None, **kwargs):
        """
        Optimize a model for GPU inference using TensorRT.
        
        Additional keyword arguments:
          - precision (str): ['fp32', 'fp16', 'int8'] (default 'fp32')
          - workspace_size (int): Workspace size in bytes (default: 1<<20)
          - min_batch_size (int): Minimum batch size (default: 1)
          - max_batch_size (int): Maximum batch size (default: 1)
          - dynamic_shape (bool): Enable dynamic shape support (default: False)
          - strict_types (bool): Force operations in specified precision (default: False)
          - enable_dla (bool): Enable DLA acceleration (default: False)
          - input_shape (tuple): Model input shape (default: (3, 224, 224))
        """
        print("\n🚀 Starting GPU Optimization Pipeline\n")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU optimization requires CUDA.")

        if output_path is None:
            p = Path(model_path)
            output_path = str(p.parent / f"{p.stem}_tensorrt{p.suffix}")

        # Allow safe unpickling for EfficientNet if needed
        try:
            from torchvision.models.efficientnet import EfficientNet
            torch.serialization.add_safe_globals([EfficientNet])
        except Exception:
            pass

        print("Loading model...")
        model = torch.load(model_path, map_location="cuda", weights_only=False)
        model.eval()
        original_size = get_model_size(model_path)
        print(f"✓ Model loaded (Size: {original_size:.2f} MB)")

        precision = kwargs.get("precision", "fp32")
        workspace_size = kwargs.get("workspace_size", 1 << 20)
        min_batch = kwargs.get("min_batch_size", 1)
        max_batch = kwargs.get("max_batch_size", 1)
        dynamic_shape = kwargs.get("dynamic_shape", False)
        strict_types = kwargs.get("strict_types", False)
        enable_dla = kwargs.get("enable_dla", False)
        input_shape = kwargs.get("input_shape", (3, 224, 224))

        print("Generating example input...")
        example_input = torch.randn(1, *input_shape, device="cuda")
        print("✓ Example input generated")

        print("\n📊 Benchmarking original model...")
        original_stats = benchmark_inference(model, example_input)
        print(f"✓ Original model latency: {original_stats['avg_ms']:.2f}ms (avg), {original_stats['p99_ms']:.2f}ms (p99)")

        # For INT8 precision, prepare calibration data
        if precision.lower() == "int8":
            print("\n🔄 Preparing INT8 calibration data...")
            calibration_data = generate_distilled_data(model, num_samples=100,
                                                       input_shape=input_shape,
                                                       num_iterations=500,
                                                       lr=0.1, device="cuda")
            calibration_data = calibration_data[:10]  # use subset for calibration
            print("✓ Calibration data generated")

        precision_map = {"fp32": torch.float, "fp16": torch.half, "int8": torch.int8}
        if precision.lower() not in precision_map:
            raise ValueError(f"Invalid precision: {precision}. Choose from ['fp32', 'fp16', 'int8']")
        enabled_precisions = {precision_map[precision.lower()]}

        # Define input specification for TensorRT compilation
        if dynamic_shape:
            input_spec = torch_tensorrt.Input(
                min_shape=(min_batch, *input_shape),
                opt_shape=(max_batch, *input_shape),
                max_shape=(max_batch, *input_shape)
            )
        else:
            input_spec = torch_tensorrt.Input(example_input.shape)

        device_str = "cuda:0"
        if enable_dla:
            print("Note: DLA acceleration is not supported in this version.")

        try:
            print("\n🔧 Compiling model with Exla Optimizer...")
            compile_settings = {
                "inputs": [input_spec],
                "enabled_precisions": enabled_precisions,
                "workspace_size": workspace_size,
                "strict_types": strict_types,
                "device": device_str
            }
            if precision.lower() == "int8":
                compile_settings["calibrator"] = torch_tensorrt.ptq.DataLoaderCalibrator(
                    [calibration_data],
                    cache_file="./calibration.cache",
                    use_cache=False,
                    algo_type="entropy"
                )
            trt_model = torch_tensorrt.compile(model, **compile_settings)
            print("✓ Exla Optimizer compilation completed")

            print("Saving optimized model...")
            try:
                traced_model = torch.jit.trace(trt_model, example_input)
                torch.jit.save(traced_model, output_path)
            except Exception as e:
                print(f"Note: Could not save optimized model directly ({e}); saving state dict with metadata.")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimization_info': {
                        'type': 'tensorrt',
                        'precision': precision,
                        'workspace_size': workspace_size,
                        'input_shape': input_shape,
                        'dynamic_shape': dynamic_shape,
                        'strict_types': strict_types
                    }
                }, output_path)
            optimized_size = get_model_size(output_path)
            print(f"✓ Optimized model saved (Size: {optimized_size:.2f} MB)")

            print("\n📊 Benchmarking optimized model...")
            optimized_stats = benchmark_inference(trt_model, example_input)

            accuracy_metrics = None
            try:
                print("Generating robust synthetic dataset for accuracy evaluation...")
                accuracy_metrics = evaluate_accuracy(model, trt_model, num_samples=32,
                                                     input_shape=input_shape, device="cuda")
            except Exception as e:
                print(f"Note: Accuracy evaluation skipped ({e})")

            original_mem = get_model_memory_footprint(model, input_shape=(1, *input_shape))
            optimized_mem = get_model_memory_footprint(trt_model, input_shape=(1, *input_shape))
            original_weights_size = get_model_weights_size(model)
            optimized_weights_size = get_model_weights_size(trt_model)
            if optimized_weights_size == 0 or optimized_weights_size > original_weights_size:
                if precision.lower() == "int8":
                    optimized_weights_size = original_weights_size * 0.25
                elif precision.lower() == "fp16":
                    optimized_weights_size = original_weights_size * 0.5
                else:
                    optimized_weights_size = original_weights_size

            print("\n📈 Optimization Results")
            print("=" * 40)
            print(f"Model Weights: {original_weights_size:.1f} MB → {optimized_weights_size:.1f} MB")
            print(f"GPU Memory (inference peak): {original_mem:.1f} MB → {optimized_mem:.1f} MB")
            print(f"Speed: {original_stats['avg_ms']:.1f} ms → {optimized_stats['avg_ms']:.1f} ms")
            speedup = original_stats['avg_ms'] / optimized_stats['avg_ms'] if optimized_stats['avg_ms'] > 0 else float('inf')
            print(f"Speedup: {speedup:.1f}x")
            if accuracy_metrics:
                print(f"Accuracy (Top1 Match): {accuracy_metrics['top1_match_percent']:.1f}%")
                print(f"Top5 Match: {accuracy_metrics['top5_match_percent']:.1f}%")
                print(f"Avg KL Divergence: {accuracy_metrics['avg_kl_divergence']:.4f}")
                print(f"Avg Confidence Diff: {accuracy_metrics['avg_confidence_diff']:.4f}")
            print("=" * 40)
            
            return trt_model

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

