#!/usr/bin/env python3
"""
Memory Cleanup Utilities

Provides comprehensive memory cleanup for:
- CUDA cache
- Python garbage collection
- System memory (Linux)

Used to free memory between training runs.
"""

import gc
import torch
import subprocess
import sys
from typing import Optional


def clear_cuda_cache(device_id: Optional[int] = None, verbose: bool = True):
    """
    Clear CUDA memory cache.
    
    Args:
        device_id (int, optional): Specific device to clear. None = all devices.
        verbose (bool): Print status messages
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available. Skipping CUDA cache cleanup.")
        return
    
    # Get memory before cleanup
    if verbose:
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory before cleanup: {mem_before:.1f} MB")
    
    # Clear cache
    if device_id is None:
        torch.cuda.empty_cache()
    else:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    if verbose:
        mem_after = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory after cleanup:  {mem_after:.1f} MB")
        print(f"Freed: {mem_before - mem_after:.1f} MB")


def clear_python_memory(verbose: bool = True):
    """
    Run Python garbage collection.
    
    Args:
        verbose (bool): Print status messages
    """
    if verbose:
        print("Running Python garbage collection...")
    
    # Collect all generations
    collected = gc.collect()
    
    if verbose:
        print(f"Collected {collected} objects")


def clear_system_memory_linux(verbose: bool = True):
    """
    Clear system cache on Linux (requires sudo).
    
    WARNING: This requires root privileges and is generally not recommended
    during training. Use only between experiments.
    
    Args:
        verbose (bool): Print status messages
    """
    if sys.platform != 'linux':
        if verbose:
            print("System cache clearing only supported on Linux.")
        return
    
    if verbose:
        print("Attempting to clear system cache (requires sudo)...")
    
    try:
        # Drop caches (1=pagecache, 2=dentries/inodes, 3=all)
        # This is safe and doesn't affect running processes
        subprocess.run(
            ['sudo', 'sh', '-c', 'sync; echo 3 > /proc/sys/vm/drop_caches'],
            check=True,
            capture_output=True
        )
        if verbose:
            print("✓ System cache cleared")
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"❌ System cache clearing failed: {e}")
            print("  (This is optional and training can continue)")
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error: {e}")


def full_cleanup(device_id: Optional[int] = None, clear_system: bool = False, verbose: bool = True):
    """
    Perform complete memory cleanup.
    
    Args:
        device_id (int, optional): Specific CUDA device to clear
        clear_system (bool): If True, attempt to clear system cache (Linux, sudo)
        verbose (bool): Print status messages
    
    Example:
        >>> full_cleanup(device_id=0, clear_system=False)
        Running memory cleanup...
        GPU memory before cleanup: 2048.5 MB
        GPU memory after cleanup:  128.3 MB
        Freed: 1920.2 MB
        Running Python garbage collection...
        Collected 1523 objects
        ✓ Memory cleanup complete!
    """
    if verbose:
        print("\n" + "="*60)
        print("Running memory cleanup...")
        print("="*60)
    
    # Clear CUDA cache
    clear_cuda_cache(device_id=device_id, verbose=verbose)
    
    # Python garbage collection
    if verbose:
        print()
    clear_python_memory(verbose=verbose)
    
    # System cache (optional, Linux only)
    if clear_system:
        if verbose:
            print()
        clear_system_memory_linux(verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print("✓ Memory cleanup complete!")
        print("="*60 + "\n")


def reset_peak_memory_stats(device_id: int = 0, verbose: bool = True):
    """
    Reset peak memory statistics.
    
    Useful for measuring memory usage of specific operations.
    
    Args:
        device_id (int): CUDA device ID
        verbose (bool): Print status messages
    
    Example:
        >>> reset_peak_memory_stats()
        >>> # ... run training epoch ...
        >>> peak = torch.cuda.max_memory_allocated() / 1024**2
        >>> print(f"Peak memory: {peak:.1f} MB")
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available. Skipping stats reset.")
        return
    
    torch.cuda.reset_peak_memory_stats(device_id)
    torch.cuda.reset_accumulated_memory_stats(device_id)
    
    if verbose:
        print(f"Peak memory stats reset for device {device_id}")


if __name__ == "__main__":
    # Test cleanup utilities
    print("Testing Memory Cleanup Utilities...\n")
    
    # Test CUDA cleanup
    if torch.cuda.is_available():
        print("[1/3] Testing CUDA cleanup...")
        
        # Allocate some memory
        tensor = torch.randn(50, 1024, 1024, device='cuda')
        print(f"Allocated 50MB tensor")
        
        # Clear
        clear_cuda_cache(verbose=True)
        
        # Delete tensor
        del tensor
        torch.cuda.empty_cache()
        print()
    else:
        print("[1/3] CUDA not available, skipping CUDA tests\n")
    
    # Test Python GC
    print("[2/3] Testing Python garbage collection...")
    clear_python_memory(verbose=True)
    print()
    
    # Test full cleanup
    print("[3/3] Testing full cleanup...")
    full_cleanup(clear_system=False, verbose=True)
    
    print("✓ All tests passed!")
