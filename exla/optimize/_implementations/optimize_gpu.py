import os
import sys
import time
from pathlib import Path

def _suppress_c_logs():
    """Suppress C-level logs by redirecting stderr to /dev/null."""
    try:
        # Save original stderr fd
        original_stderr_fd = sys.stderr.fileno()
        # Open /dev/null for writing
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Replace stderr with devnull
        os.dup2(devnull, 2)
        # Close devnull (the duplicate still exists)
        os.close(devnull)
        return original_stderr_fd
    except Exception:
        return None

# Suppress C-level logs before any imports
original_stderr = _suppress_c_logs()

# Now import remaining modules
import torch_tensorrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import OptimizeBase

# Utility function: Simple progress spinner.
def print_progress(message, progress=None):
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    sp = spinner[int(time.time() * 10) % len(spinner)]
    if progress is not None:
        print(f"\r{sp} {message} ({progress:.1f}%)", end="", flush=True)
    else:
        print(f"\r{sp} {message}", end="", flush=True)

# Utility function: Return file size in MB.
def get_model_size(model_or_path):
    try:
        if isinstance(model_or_path, str):
            return Path(model_or_path).stat().st_size / (1024 * 1024)
        tmp_path = "tmp_model.pt"
        try:
            torch.save(model_or_path, tmp_path)
        except Exception:
            torch.save(model_or_path.state_dict(), tmp_path)
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        os.remove(tmp_path)
        return size_mb
    except Exception:
        return 0

# Utility function: Benchmark inference latency.
def benchmark_inference(model, input_tensor, num_warmup=5, num_iters=100):
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
    print()
    sorted_lat = sorted(latencies)
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p90_ms": sorted_lat[int(0.9 * len(latencies))],
        "p99_ms": sorted_lat[int(0.99 * len(latencies))],
        "min_ms": min(latencies),
        "max_ms": max(latencies)
    }

# Utility function: Compute a loss based on BatchNorm statistics.
def bn_stats_loss(model, input_tensor):
    total_loss = torch.tensor(0.0, device=input_tensor.device, requires_grad=True)
    hooks = []
    def hook_fn(module, inp, out):
        nonlocal total_loss
        out_clone = out.clone()
        dims = [0, 2, 3]
        act_mean = out_clone.mean(dim=dims)
        act_var = ((out_clone - act_mean.view(1, -1, 1, 1)) ** 2).mean(dim=dims)
        running_mean = module.running_mean.to(act_mean.device)
        running_var = module.running_var.to(act_var.device)
        total_loss = total_loss + F.mse_loss(act_mean, running_mean.detach()) + F.mse_loss(act_var, running_var.detach())
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))
    with torch.set_grad_enabled(True):
        _ = model(input_tensor)
    for hook in hooks:
        hook.remove()
    return total_loss

# Utility function: Generate synthetic data matching BN statistics.
def generate_distilled_data(model, num_samples=50, input_shape=(3, 224, 224), num_iters=100, lr=0.1, device="cuda"):
    data = torch.randn(num_samples, *input_shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([data], lr=lr)
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = bn_stats_loss(model, data)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            data.clamp_(-1, 1)
        if i % 10 == 0:
            print_progress("Generating synthetic data", (i + 1) * 100 / num_iters)
    print()
    return data.detach()

# Utility function: Evaluate accuracy by comparing predictions.
def evaluate_accuracy(orig_model, opt_model, num_samples=32, input_shape=(3, 224, 224), device="cuda"):
    print("Generating synthetic dataset for accuracy evaluation...")
    data = generate_distilled_data(orig_model, num_samples=num_samples, input_shape=input_shape,
                                   num_iters=200, lr=0.05, device=device).to(device)
    top1, top5 = 0, 0
    for i in range(num_samples):
        sample = data[i].unsqueeze(0)
        with torch.no_grad():
            o_logits = orig_model(sample)
            opt_logits = opt_model(sample)
        o_prob = F.softmax(o_logits, dim=1)[0]
        opt_prob = F.softmax(opt_logits, dim=1)[0]
        if torch.argmax(o_prob) == torch.argmax(opt_prob):
            top1 += 1
        if any(cls in opt_prob.topk(5)[1].tolist() for cls in o_prob.topk(5)[1].tolist()):
            top5 += 1
    return {"top1": top1 / num_samples * 100, "top5": top5 / num_samples * 100}

# Utility function: Measure GPU memory usage (in MB) during inference.
def get_model_memory_footprint(model, input_shape=(1, 3, 224, 224), device="cuda"):
    measurements = []
    for _ in range(5):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(input_shape, device=device)
        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize()
        measurements.append(torch.cuda.max_memory_allocated(device))
        del x
        torch.cuda.empty_cache()
    avg_peak = sum(measurements) / len(measurements)
    return avg_peak / (1024 * 1024)

# Utility function: Compute the total size of model weights (in MB).
def get_model_weights_size(model):
    try:
        # First try to get size from state_dict
        size = 0
        try:
            for param in model.state_dict().values():
                size += param.nelement() * param.element_size()
            return size / (1024 * 1024)
        except Exception:
            # If state_dict fails (e.g. for TensorRT models), try saving the model
            tmp_path = "tmp_model.pt"
            try:
                torch.save(model, tmp_path)
                size = Path(tmp_path).stat().st_size / (1024 * 1024)
                os.remove(tmp_path)
                return size
            except Exception:
                # If direct save fails, try to save traced model
                traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224, device="cuda"))
                torch.jit.save(traced, tmp_path)
                size = Path(tmp_path).stat().st_size / (1024 * 1024)
                os.remove(tmp_path)
                return size
    except Exception:
        return 0

# GPU Optimization class.
class OptimizeGPU(OptimizeBase):
    def optimize(self, model_path: str, output_path: str = None, **kwargs):
        print("\n🚀 Starting GPU Optimization Pipeline\n")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if output_path is None:
            p = Path(model_path)
            output_path = str(p.parent / f"{p.stem}_tensorrt{p.suffix}")

        # Allow safe unpickling for EfficientNet.
        try:
            from torchvision.models.efficientnet import EfficientNet
            torch.serialization.add_safe_globals([EfficientNet])
        except Exception:
            pass

        print("Loading model...")
        model = torch.load(model_path, map_location="cuda", weights_only=False)
        model.eval()
        orig_size = get_model_size(model_path)
        print(f"✓ Model loaded (Size: {orig_size:.2f} MB)")

        precision = kwargs.get("precision", "fp32")
        ws = kwargs.get("workspace_size", 1 << 20)
        dynamic_shape = kwargs.get("dynamic_shape", False)
        input_shape = kwargs.get("input_shape", (3, 224, 224))

        print("Generating example input...")
        ex_input = torch.randn(1, *input_shape, device="cuda")
        print("✓ Example input generated")

        print("\n📊 Benchmarking original model...")
        orig_stats = benchmark_inference(model, ex_input)
        print(f"✓ Original model latency: {orig_stats['avg_ms']:.2f}ms (avg), {orig_stats['p99_ms']:.2f}ms (p99)")

        if precision.lower() == "int8":
            print("\n🔄 Preparing INT8 calibration data...")
            cal_data = generate_distilled_data(model, num_samples=100, input_shape=input_shape,
                                               num_iters=500, lr=0.1, device="cuda")
            cal_data = cal_data[:10]
            print("✓ Calibration data generated")

        prec_map = {"fp32": torch.float, "fp16": torch.half, "int8": torch.int8}
        if precision.lower() not in prec_map:
            raise ValueError("Invalid precision.")
        enabled_precisions = {prec_map[precision.lower()]}

        if dynamic_shape:
            input_spec = torch_tensorrt.Input(min_shape=(1, *input_shape),
                                              opt_shape=(1, *input_shape),
                                              max_shape=(1, *input_shape))
        else:
            input_spec = torch_tensorrt.Input(ex_input.shape)
        device_str = "cuda:0"

        try:
            print("\n🔧 Compiling model with Exla Optimizer...")
            compile_settings = {
                "inputs": [input_spec],
                "enabled_precisions": enabled_precisions,
                "workspace_size": ws,
                "device": device_str
            }
            if precision.lower() == "int8":
                compile_settings["calibrator"] = torch_tensorrt.ptq.DataLoaderCalibrator(
                    [cal_data],
                    cache_file="./calibration.cache",
                    use_cache=False,
                    algo_type="entropy"
                )
            trt_model = torch_tensorrt.compile(model, **compile_settings)
            print("✓ Exla Optimizer compilation completed")

            print("Saving optimized model...")
            try:
                traced = torch.jit.trace(trt_model, ex_input)
                torch.jit.save(traced, output_path)
            except Exception as e:
                print(f"Note: Could not save optimized model directly ({e}); saving state dict.")
                torch.save({"model_state_dict": model.state_dict()}, output_path)
            opt_size = get_model_size(output_path)
            print(f"✓ Optimized model saved (Size: {opt_size:.2f} MB)")

            print("\n📊 Benchmarking optimized model...")
            opt_stats = benchmark_inference(trt_model, ex_input)
            acc = None
            try:
                print("Generating synthetic dataset for accuracy evaluation...")
                acc = evaluate_accuracy(model, trt_model, num_samples=32, input_shape=input_shape, device="cuda")
            except Exception as e:
                print(f"Note: Accuracy evaluation skipped ({e})")
            mem_orig = get_model_memory_footprint(model, input_shape=(1, *input_shape))
            mem_opt = get_model_memory_footprint(trt_model, input_shape=(1, *input_shape))
            wt_orig = get_model_weights_size(model)
            wt_opt = get_model_weights_size(trt_model)

            print("\n📈 Optimization Results")
            print("=" * 40)
            print(f"Model Weights: {wt_orig:.1f} MB → {wt_opt:.1f} MB")
            print(f"GPU Memory (peak): {mem_orig:.1f} MB → {mem_opt:.1f} MB")
            print(f"Speed: {orig_stats['avg_ms']:.1f} ms → {opt_stats['avg_ms']:.1f} ms")
            speedup = orig_stats['avg_ms'] / opt_stats['avg_ms'] if opt_stats['avg_ms'] > 0 else float('inf')
            print(f"Speedup: {speedup:.1f}x")
            if acc:
                print(f"Accuracy (Top1): {acc['top1']:.1f}%, Top5: {acc['top5']:.1f}%")
            print("=" * 40)
            return trt_model
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")
