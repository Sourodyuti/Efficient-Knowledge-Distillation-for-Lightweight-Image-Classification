#!/bin/bash
# CNN Distillation Memory Cleanup Script
# Clears CUDA cache, peak memory stats, Python GC, and optionally system cache

echo "======================================"
echo "CNN Distillation - Memory Cleanup"
echo "======================================"

# Navigate to script directory
cd "$(dirname "$0")" || exit

# Add parent directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"

echo ""
echo "[1/4] Clearing CUDA cache..."
python3 << EOF
import sys
sys.path.insert(0, '..')
from src.utils.memory_cleanup import clear_cuda_cache
clear_cuda_cache(verbose=True)
EOF

echo ""
echo "[2/4] Resetting peak memory stats..."
python3 << EOF
import torch
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    print("\u2713 Peak memory stats reset")
else:
    print("No CUDA device available")
EOF

echo ""
echo "[3/4] Running Python garbage collection..."
python3 << EOF
import sys
sys.path.insert(0, '..')
from src.utils.memory_cleanup import clear_python_memory
clear_python_memory(verbose=True)
EOF

echo ""
echo "[4/4] System cache (optional - requires sudo)..."
read -p "Clear system cache? This requires sudo (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    echo "✓ System cache cleared"
else
    echo "Skipping system cache cleanup"
fi

echo ""
echo "======================================"
echo "✓ Memory cleanup complete!"
echo "======================================"

echo ""
echo "Memory usage after cleanup:"
python3 << EOF
import sys
sys.path.insert(0, '..')
from src.utils.gpu_monitor import GPUMonitor
import torch

if torch.cuda.is_available():
    monitor = GPUMonitor(device_id=0)
    monitor.print_summary()
else:
    print("No CUDA device available")
EOF
