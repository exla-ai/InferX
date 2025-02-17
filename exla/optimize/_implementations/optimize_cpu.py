from ._base import OptimizeBase
import torch
from pathlib import Path
import os
import time
import torch.nn as nn
import torch.nn.functional as F

class OptimizeCPU(OptimizeBase):
    def optimize(self, model_path: str, **kwargs):
        """
        Optimize model for CPU inference using dynamic quantization.
        
        Args:
            model_path (str): Path to input model
            **kwargs: Additional optimization parameters
                - num_threads (int): Number of CPU threads to use
                - quantize (bool): Whether to quantize the model (default: True)
                - quantize_dtype (torch.dtype): Quantization data type (default: torch.qint8)
                - optimization_level (str): One of ['speed', 'memory', 'balanced'] (default: 'balanced')
        """
        print(f"Optimizing model for CPU: {model_path}")
        
        # Get optimization parameters
        num_threads = kwargs.get('num_threads', torch.get_num_threads())
        quantize = kwargs.get('quantize', True)
        quantize_dtype = kwargs.get('quantize_dtype', torch.qint8)
        opt_level = kwargs.get('optimization_level', 'balanced')
        
        print(f"CPU Optimization parameters:")
        print(f"- Threads: {num_threads}")
        print(f"- Quantization: {quantize}")
        print(f"- Quantization dtype: {quantize_dtype}")
        print(f"- Optimization level: {opt_level}")
        
        overall_start = time.time()
        
        # Load and prepare model
        print("Loading model...")
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        
        # Disable in-place operations
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
            if isinstance(module, (nn.ReLU, nn.SiLU, nn.LeakyReLU, nn.ELU, nn.CELU, nn.SELU, nn.RReLU)):
                module.inplace = False
        
        # Set number of threads
        torch.set_num_threads(num_threads)
        
        # Estimate original model memory
        temp_orig = "temp_orig.pt"
        torch.save(model.state_dict(), temp_orig)
        orig_mem = os.path.getsize(temp_orig)
        
        # Generate synthetic data for evaluation
        print("Generating synthetic dataset...")
        synthetic_data = generate_distilled_data(model, num_samples=100, input_shape=(3,224,224),
                                               num_iterations=500, lr=0.1, device="cpu")
        
        # Evaluate original model
        orig_acc = self._evaluate_model(model, synthetic_data, device="cpu")
        print(f"Original model accuracy on synthetic data: {orig_acc:.2f}%")
        
        # Apply quantization if requested
        conv_start = time.time()
        if quantize:
            print(f"Applying dynamic quantization (dtype: {quantize_dtype})...")
            model_optimized = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d},  # Quantize both linear and conv layers
                dtype=quantize_dtype
            )
        else:
            model_optimized = model
            
        conv_end = time.time()
        conversion_time = conv_end - conv_start
        
        # Fine-tune if quantized
        if quantize:
            print("Fine-tuning quantized model...")
            model_optimized = fine_tune_quantized(model, model_optimized, synthetic_data,
                                                epochs=5, batch_size=10, lr=1e-4,
                                                device="cpu")
        
        # Evaluate optimized model
        opt_acc = self._evaluate_model(model_optimized, synthetic_data, device="cpu")
        print(f"Optimized model accuracy on synthetic data: {opt_acc:.2f}%")
        
        # Benchmark inference speed
        dummy_input = torch.randn(1, 3, 224, 224)
        orig_time = self._benchmark_model(model, dummy_input, device="cpu", n_runs=50)
        opt_time = self._benchmark_model(model_optimized, dummy_input, device="cpu", n_runs=50)
        avg_orig_time = sum(orig_time)/len(orig_time)
        avg_opt_time = sum(opt_time)/len(opt_time)
        speedup = avg_orig_time / avg_opt_time if avg_opt_time > 0 else float('inf')
        
        # Estimate optimized model memory
        temp_opt = "temp_opt.pt"
        torch.save(model_optimized.state_dict(), temp_opt)
        opt_mem = os.path.getsize(temp_opt)
        mem_reduction = ((orig_mem - opt_mem) / orig_mem) * 100
        os.remove(temp_orig)
        os.remove(temp_opt)
        
        overall_time = time.time() - overall_start
        
        print("\n==== CPU Optimization Summary ====")
        print(f"Conversion time: {conversion_time:.3f} seconds")
        print(f"Original model memory: {orig_mem/1e6:.2f} MB, Optimized model memory: {opt_mem/1e6:.2f} MB, Reduction: {mem_reduction:.2f}%")
        print(f"Accuracy on synthetic data - Original: {orig_acc:.2f}%, Optimized: {opt_acc:.2f}%")
        print(f"Average inference time (ms) - Original: {avg_orig_time:.3f}, Optimized: {avg_opt_time:.3f}, Speedup: {speedup:.2f}x")
        print(f"Total optimization time: {overall_time:.3f} seconds")
        
        # Save the optimized model
        p = Path(model_path)
        output_path = str(p.parent / f"{p.stem}_cpu_optimized{p.suffix}")
        torch.save(model_optimized.state_dict(), output_path)
        print(f"Optimized model saved to: {output_path}")
        
        return {
            "original_path": model_path,
            "optimized_path": output_path,
            "device": "cpu",
            "original_accuracy_percent": orig_acc,
            "optimized_accuracy_percent": opt_acc,
            "conversion_time_sec": conversion_time,
            "original_memory_bytes": orig_mem,
            "optimized_memory_bytes": opt_mem,
            "memory_reduction_percent": mem_reduction,
            "avg_inference_time_original_ms": avg_orig_time,
            "avg_inference_time_optimized_ms": avg_opt_time,
            "speedup": speedup,
            "total_optimization_time_sec": overall_time,
            "optimization_params": {
                "num_threads": num_threads,
                "quantize": quantize,
                "quantize_dtype": str(quantize_dtype),
                "optimization_level": opt_level
            }
        }