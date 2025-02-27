import os
import subprocess
import psutil
import platform
from .device_detect import detect_device, is_jetson_device

class ResourceMonitor:
    """
    Utility class to monitor system resources like GPU VRAM and CPU memory
    across different platforms (Jetson, standard NVIDIA GPU, CPU-only).
    """
    
    def __init__(self):
        """Initialize the resource monitor with device detection."""
        self.device_info = detect_device()
        self.device_type = self.device_info['type']
        self.is_gpu_available = self.device_info['capabilities']['gpu_available']
        self.is_jetson = is_jetson_device()
    
    def get_memory_usage(self, print_info=True):
        """
        Get memory usage information for the current device.
        
        Args:
            print_info (bool): Whether to print the information
            
        Returns:
            dict: Memory usage information
        """
        result = {
            'device_type': self.device_type,
            'cpu_memory': self._get_cpu_memory(),
        }
        
        # Add GPU memory info if available
        if self.is_gpu_available:
            gpu_memory = self._get_gpu_memory()
            if gpu_memory:
                result['gpu_memory'] = gpu_memory
        
        # Print information if requested
        if print_info:
            self._print_memory_info(result)
            
        return result
    
    def _get_cpu_memory(self):
        """Get CPU memory usage information."""
        vm = psutil.virtual_memory()
        return {
            'total': self._format_bytes(vm.total),
            'used': self._format_bytes(vm.used),
            'percent': vm.percent
        }
    
    def _get_gpu_memory(self):
        """Get GPU memory usage based on device type."""
        if self.is_jetson:
            return self._get_jetson_memory()
        else:
            return self._get_nvidia_memory()
    
    def _get_jetson_memory(self):
        """Get memory usage specifically for Jetson devices."""
        try:
            # Try using tegrastats for Jetson devices
            if os.path.exists('/usr/bin/tegrastats'):
                output = subprocess.check_output(
                    ['/usr/bin/tegrastats', '--interval', '1', '--count', '1'],
                    universal_newlines=True
                )
                
                # Parse RAM usage
                ram_info = None
                for part in output.split():
                    if part.startswith('RAM'):
                        ram_parts = part.split('=')[1].split('/')
                        used_mb = int(ram_parts[0][:-2])
                        total_mb = int(ram_parts[1][:-2])
                        percent = (used_mb / total_mb) * 100
                        ram_info = {
                            'total': f"{total_mb}MB",
                            'used': f"{used_mb}MB",
                            'percent': round(percent, 1)
                        }
                
                # Parse GPU memory if available
                gpu_info = None
                for part in output.split():
                    if part.startswith('GR3D_FREQ'):
                        # This indicates GPU is active
                        # Try to find memory info
                        for mem_part in output.split():
                            if mem_part.startswith('IRAM') or mem_part.startswith('VIC'):
                                mem_parts = mem_part.split('=')[1].split('/')
                                used_mb = int(mem_parts[0][:-2])
                                total_mb = int(mem_parts[1][:-2])
                                percent = (used_mb / total_mb) * 100
                                gpu_info = {
                                    'total': f"{total_mb}MB",
                                    'used': f"{used_mb}MB",
                                    'percent': round(percent, 1)
                                }
                                break
                
                return {
                    'ram': ram_info,
                    'gpu': gpu_info
                }
            
            # Fallback to nvidia-smi if tegrastats is not available
            return self._get_nvidia_memory()
            
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError) as e:
            print(f"Warning: Could not get Jetson memory info: {e}")
            return None
    
    def _get_nvidia_memory(self):
        """Get memory usage for standard NVIDIA GPUs using nvidia-smi."""
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                universal_newlines=True
            )
            
            # Parse the output
            values = output.strip().split(',')
            total_mb = int(values[0].strip())
            used_mb = int(values[1].strip())
            free_mb = int(values[2].strip())
            gpu_util = int(values[3].strip())
            
            return {
                'total': f"{total_mb}MB",
                'used': f"{used_mb}MB",
                'free': f"{free_mb}MB",
                'percent': round((used_mb / total_mb) * 100, 1),
                'utilization': f"{gpu_util}%"
            }
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError) as e:
            print(f"Warning: Could not get NVIDIA GPU memory info: {e}")
            return None
    
    def _format_bytes(self, bytes_value):
        """Format bytes to a human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f}PB"
    
    def _print_memory_info(self, info):
        """Print memory information in a formatted way."""
        device_name = {
            'cpu': 'CPU',
            'gpu': 'NVIDIA GPU',
            'agx_orin': 'NVIDIA Jetson AGX Orin',
            'orin_nano': 'NVIDIA Jetson Orin Nano'
        }.get(info['device_type'], info['device_type'].upper())
        
        print(f"\n📊 Resource Monitor - {device_name}")
        
        # Print CPU memory
        cpu_mem = info['cpu_memory']
        print(f"💻 System Memory: {cpu_mem['used']} / {cpu_mem['total']} ({cpu_mem['percent']}%)")
        
        # Print GPU memory if available
        if 'gpu_memory' in info:
            gpu_mem = info['gpu_memory']
            
            if self.is_jetson and 'ram' in gpu_mem and gpu_mem['ram']:
                print(f"🧠 Jetson RAM: {gpu_mem['ram']['used']} / {gpu_mem['ram']['total']} ({gpu_mem['ram']['percent']}%)")
                
            if self.is_jetson and 'gpu' in gpu_mem and gpu_mem['gpu']:
                print(f"🎮 Jetson GPU Memory: {gpu_mem['gpu']['used']} / {gpu_mem['gpu']['total']} ({gpu_mem['gpu']['percent']}%)")
            elif not self.is_jetson and gpu_mem:
                print(f"🎮 GPU Memory: {gpu_mem['used']} / {gpu_mem['total']} ({gpu_mem['percent']}%)")
                print(f"   GPU Utilization: {gpu_mem['utilization']}")
        else:
            print("🎮 GPU: Not available")

# Convenience function
def print_resource_usage():
    """Print current resource usage."""
    monitor = ResourceMonitor()
    return monitor.get_memory_usage(print_info=True)

# Example usage
if __name__ == "__main__":
    print_resource_usage() 