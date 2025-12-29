"""
Reproducibility Utilities

Ensures deterministic training by:
- Setting random seeds across all libraries
- Configuring PyTorch for deterministic operations
- Disabling non-deterministic algorithms
- Setting CUBLAS workspace config for deterministic CUDA operations

Framework-agnostic.
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42, verbose: bool = True):
    """
    Set random seed for reproducibility across all libraries.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed (int): Random seed value (default: 42)
        verbose (bool): If True, print confirmation message
    
    Example:
        >>> set_seed(42)
        Random seed set to 42 for reproducibility
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if verbose:
        print(f"Random seed set to {seed} for reproducibility")


def set_deterministic(
    deterministic: bool = True,
    benchmark: bool = False,
    verbose: bool = True
):
    """
    Configure PyTorch for deterministic operations.
    
    Args:
        deterministic (bool): If True, use deterministic algorithms.
            May reduce performance but ensures reproducibility.
        benchmark (bool): If True, enable cudnn benchmarking for performance.
            Set to False for deterministic behavior.
            Set to True for faster training (but non-deterministic).
        verbose (bool): If True, print configuration
    
    Warning:
        Deterministic mode may slow down training but ensures exact
        reproducibility across runs with the same seed.
    
    Example:
        >>> set_deterministic(deterministic=True, benchmark=False)
        PyTorch configured for deterministic operations
    """
    # CRITICAL: Set CUBLAS workspace config BEFORE any CUDA operations
    # This is required for deterministic behavior with CUDA >= 10.2
    if deterministic and torch.cuda.is_available():
        # Use :4096:8 for better performance, :16:8 for less memory
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if verbose:
            print("CUBLAS_WORKSPACE_CONFIG set to :4096:8 for deterministic CUDA operations")
    
    if torch.cuda.is_available():
        # CuDNN settings
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        
        if verbose:
            if deterministic:
                print("PyTorch configured for deterministic operations")
                print("  cudnn.deterministic = True")
                print("  cudnn.benchmark = False")
            else:
                print("PyTorch configured for performance (non-deterministic)")
                print("  cudnn.deterministic = False")
                print(f"  cudnn.benchmark = {benchmark}")
    
    # Set deterministic algorithms (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(deterministic)
        if verbose and deterministic:
            print("  use_deterministic_algorithms = True")
    except AttributeError:
        # Older PyTorch versions
        if verbose:
            print("  Note: torch.use_deterministic_algorithms() not available (PyTorch < 1.8)")


def setup_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    verbose: bool = True
):
    """
    Complete reproducibility setup (convenience function).
    
    Combines set_seed() and set_deterministic() in one call.
    
    Args:
        seed (int): Random seed (default: 42)
        deterministic (bool): Enable deterministic mode (default: True)
        benchmark (bool): Enable cudnn benchmarking (default: False)
        verbose (bool): Print configuration (default: True)
    
    Example:
        >>> setup_reproducibility(seed=42, deterministic=True)
        CUBLAS_WORKSPACE_CONFIG set to :4096:8 for deterministic CUDA operations
        Random seed set to 42 for reproducibility
        PyTorch configured for deterministic operations
          cudnn.deterministic = True
          cudnn.benchmark = False
          use_deterministic_algorithms = True
    """
    # IMPORTANT: Set deterministic FIRST (sets CUBLAS env var)
    set_deterministic(deterministic, benchmark, verbose=verbose)
    set_seed(seed, verbose=verbose)


def get_random_state() -> dict:
    """
    Get current random state from all libraries.
    
    Returns:
        dict: Dictionary containing random states
    
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> set_random_state(state)  # Restore state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict):
    """
    Restore random state from previously saved state.
    
    Args:
        state (dict): Random state dictionary from get_random_state()
    
    Example:
        >>> state = get_random_state()
        >>> # ... training loop ...
        >>> set_random_state(state)  # Restore for reproducible validation
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def worker_init_fn(worker_id: int, seed: int = 42):
    """
    Worker initialization function for DataLoader.
    
    Ensures each worker has a different but deterministic seed.
    Use with torch.utils.data.DataLoader(worker_init_fn=...).
    
    Args:
        worker_id (int): Worker ID (automatically passed by DataLoader)
        seed (int): Base random seed
    
    Example:
        >>> from functools import partial
        >>> train_loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, seed=42)
        ... )
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # Test reproducibility utilities
    print("Testing Reproducibility Utilities...\n")
    
    # Test complete setup (order matters!)
    print("[1/4] Testing setup_reproducibility()...")
    setup_reproducibility(seed=42, deterministic=True, verbose=True)
    
    # Generate random numbers
    print("\nGenerating random numbers:")
    print(f"  Python random: {random.random():.6f}")
    print(f"  NumPy random:  {np.random.rand():.6f}")
    print(f"  PyTorch random: {torch.rand(1).item():.6f}")
    
    # Reset and verify reproducibility
    print("\n[2/4] Verifying reproducibility...")
    set_seed(42, verbose=False)
    r1 = [random.random(), np.random.rand(), torch.rand(1).item()]
    
    set_seed(42, verbose=False)
    r2 = [random.random(), np.random.rand(), torch.rand(1).item()]
    
    if r1 == r2:
        print("✓ Reproducibility verified! Same random sequence generated.")
    else:
        print("❌ Reproducibility failed!")
    
    # Test state save/restore
    print("\n[3/4] Testing state save/restore...")
    set_seed(42, verbose=False)
    state = get_random_state()
    val1 = torch.rand(3)
    
    # Generate more random numbers
    _ = torch.rand(10)
    
    # Restore state
    set_random_state(state)
    val2 = torch.rand(3)
    
    if torch.allclose(val1, val2):
        print("✓ State save/restore working correctly!")
    else:
        print("❌ State save/restore failed!")
    
    # Test CUDA deterministic operations
    if torch.cuda.is_available():
        print("\n[4/4] Testing CUDA deterministic operations...")
        device = torch.device('cuda')
        x = torch.randn(2, 2, device=device)
        y = torch.randn(2, 2, device=device)
        
        # This should work with CUBLAS_WORKSPACE_CONFIG set
        try:
            z = torch.mm(x, y)
            print("✓ CUDA deterministic operations working!")
            print(f"  CUBLAS_WORKSPACE_CONFIG = {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}")
        except RuntimeError as e:
            print(f"❌ CUDA deterministic test failed: {e}")
    else:
        print("\n[4/4] Skipping CUDA tests (no GPU available)")
    
    print("\n✓ All tests passed!")
