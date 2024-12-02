#!/bin/bash

# Check TensorFlow installation
echo "Checking TensorFlow installation..."

python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f'TensorFlow GPU available: Yes')
    print(f'Number of GPUs: {len(gpus)}')
else:
    print(f'TensorFlow GPU available: No')
"

