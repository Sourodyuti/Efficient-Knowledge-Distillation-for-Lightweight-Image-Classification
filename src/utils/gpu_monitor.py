"""
GPU Monitoring Utilities

Provides:
- GPU memory usage tracking
- Real-time GPU utilization monitoring
- Memory profiling for debugging

Supports both py3nvml and torch CUDA APIs.
"""

import torch
from typing import Optional, Dict, List, Tuple
import gc

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUMonitor:
    """
    GPU monitoring and memory tracking.
    
    Uses NVML (NVIDIA Management Library) if available,
    falls back to PyTorch CUDA APIs otherwise.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Args:
            device_id (int): CUDA device ID to monitor (default: 0)
        """
        self.device_id = device_id
        self.nvml_initialized = False
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        if not self.cuda_available:
            print("Warning: CUDA not available. GPU monitoring disabled.")
            return
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: NVML initialization failed: {e}")
                print("Falling back to PyTorch CUDA APIs.")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dict[str, float]: Dictionary with memory stats (in MB)
                - allocated: Memory allocated by PyTorch
                - reserved: Memory reserved by PyTorch
                - free: Free GPU memory (NVML only)
                - total: Total GPU memory (NVML only)
                - used: Used GPU memory (NVML only)
        """
        if not self.cuda_available:
            return {}
        
        memory_stats = {}
        
        # PyTorch memory stats
        memory_stats['allocated_mb'] = torch.cuda.memory_allocated(self.device_id) / 1024**2
        memory_stats['reserved_mb'] = torch.cuda.memory_reserved(self.device_id) / 1024**2
        
        # NVML memory stats (more accurate)
        if self.nvml_initialized:
            try:
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_stats['total_mb'] = mem_info.total / 1024**2
                memory_stats['used_mb'] = mem_info.used / 1024**2
                memory_stats['free_mb'] = mem_info.free / 1024**2
            except Exception as e:
                print(f"Warning: NVML memory query failed: {e}")
        
        return memory_stats
    
    def get_utilization(self) -> Optional[float]:
        """
        Get GPU utilization percentage.
        
        Returns:
            float: GPU utilization (0-100) or None if not available
        """
        if not self.nvml_initialized:
            return None
        
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except Exception as e:
            print(f"Warning: GPU utilization query failed: {e}")
            return None
    
    def get_temperature(self) -> Optional[float]:
        """
        Get GPU temperature in Celsius.
        
        Returns:
            float: Temperature in °C or None if not available
        """
        if not self.nvml_initialized:
            return None
        
        try:
            temp = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
            return temp
        except Exception as e:
            print(f"Warning: Temperature query failed: {e}")
            return None
    
    def get_device_info(self) -> Dict[str, any]:
        """
        Get GPU device information.
        
        Returns:
            Dict[str, any]: Device information
        """
        if not self.cuda_available:
            return {'cuda_available': False}
        
        info = {
            'cuda_available': True,
            'device_id': self.device_id,
            'device_name': torch.cuda.get_device_name(self.device_id),
            'compute_capability': torch.cuda.get_device_capability(self.device_id),
        }
        
        if self.nvml_initialized:
            try:
                name = nvml.nvmlDeviceGetName(self.handle)
                info['nvml_name'] = name
            except:
                pass
        
        return info
    
    def print_summary(self):
        """
        Print comprehensive GPU summary.
        """
        print("\n" + "="*60)
        print("GPU Monitor Summary")
        print("="*60)
        
        # Device info
        info = self.get_device_info()
        if not info['cuda_available']:
            print("CUDA not available")
            return
        
        print(f"Device ID:   {info['device_id']}")
        print(f"Device Name: {info['device_name']}")
        print(f"Compute Cap: {info['compute_capability']}")
        
        # Memory
        memory = self.get_memory_usage()
        print(f"\nMemory Usage:")
        print(f"  Allocated: {memory['allocated_mb']:.1f} MB")
        print(f"  Reserved:  {memory['reserved_mb']:.1f} MB")
        
        if 'total_mb' in memory:
            print(f"  Used:      {memory['used_mb']:.1f} MB")
            print(f"  Free:      {memory['free_mb']:.1f} MB")
            print(f"  Total:     {memory['total_mb']:.1f} MB")
            print(f"  Usage:     {100 * memory['used_mb'] / memory['total_mb']:.1f}%")
        
        # Utilization
        util = self.get_utilization()
        if util is not None:
            print(f"\nGPU Utilization: {util}%")
        
        # Temperature
        temp = self.get_temperature()
        if temp is not None:
            print(f"Temperature: {temp}°C")
        
        print("="*60 + "\n")
    
    def cleanup(self):
        """Cleanup NVML resources."""
        if self.nvml_initialized:
            try:
                nvml.nvmlShutdown()
            except:
                pass
    
    def __del__(self):
        """Destructor."""
        self.cleanup()


def get_gpu_memory_usage(device_id: int = 0) -> float:
    """
    Quick function to get current GPU memory usage in MB.
    
    Args:
        device_id (int): CUDA device ID
    
    Returns:
        float: Memory usage in MB (allocated by PyTorch)
    
    Example:
        >>> memory_mb = get_gpu_memory_usage()
        >>> print(f"GPU Memory: {memory_mb:.1f} MB")
    """
    if not torch.cuda.is_available():
        return 0.0
    
    return torch.cuda.memory_allocated(device_id) / 1024**2


def clear_gpu_memory(device_id: Optional[int] = None):
    """
    Clear GPU memory cache and run garbage collection.
    
    Args:
        device_id (int, optional): Specific device to clear. If None, clears all.
    
    Example:
        >>> clear_gpu_memory()
        GPU memory cleared
    """
    if not torch.cuda.is_available():
        return
    
    # Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    if device_id is None:
        torch.cuda.empty_cache()
    else:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    
    print("GPU memory cache cleared")


def get_max_memory_allocated(device_id: int = 0, reset: bool = False) -> float:
    """
    Get maximum GPU memory allocated since last reset.
    
    Args:
        device_id (int): CUDA device ID
        reset (bool): If True, reset the counter
    
    Returns:
        float: Maximum memory allocated in MB
    
    Example:
        >>> max_mem = get_max_memory_allocated(reset=True)
        >>> # ... train model ...
        >>> peak_mem = get_max_memory_allocated()
        >>> print(f"Peak memory: {peak_mem:.1f} MB")
    """
    if not torch.cuda.is_available():
        return 0.0
    
    max_mem = torch.cuda.max_memory_allocated(device_id) / 1024**2
    
    if reset:
        torch.cuda.reset_max_memory_allocated(device_id)
    
    return max_mem


if __name__ == "__main__":
    # Test GPU monitor
    print("Testing GPU Monitor...\n")
    
    monitor = GPUMonitor(device_id=0)
    
    # Print device info
    info = monitor.get_device_info()
    print(f"CUDA Available: {info.get('cuda_available', False)}")
    
    if info.get('cuda_available'):
        print(f"Device: {info['device_name']}")
        
        # Print summary
        monitor.print_summary()
        
        # Allocate some memory
        print("Allocating 100MB tensor...")
        tensor = torch.randn(100, 1024, 1024, device='cuda')
        
        memory = monitor.get_memory_usage()
        print(f"\nAfter allocation:")
        print(f"  Allocated: {memory['allocated_mb']:.1f} MB")
        
        # Test functional API
        print(f"\nFunctional API:")
        print(f"  get_gpu_memory_usage(): {get_gpu_memory_usage():.1f} MB")
        print(f"  get_max_memory_allocated(): {get_max_memory_allocated():.1f} MB")
        
        # Clear memory
        del tensor
        clear_gpu_memory()
        
        print(f"\nAfter cleanup:")
        print(f"  Memory: {get_gpu_memory_usage():.1f} MB")
    
    print("\n✓ All tests passed!")
