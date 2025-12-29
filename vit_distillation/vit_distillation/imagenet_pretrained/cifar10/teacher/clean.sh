#!/bin/bash
echo "Cleaning up vit_distillation/imagenet_pretrained/cifar10/teacher..."
python3 << EOF
import sys
sys.path.insert(0, '/home/souro/Downloads/Efficient-Knowledge-Distillation-for-Lightweight-Image-Classification')
from src.utils.memory_cleanup import full_cleanup
full_cleanup(device_id=0, clear_system=False, verbose=True)
EOF
echo "Cleanup complete!"
