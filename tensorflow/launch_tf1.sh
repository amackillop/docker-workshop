docker run -it --rm --name tf1 --gpus "device=0" tensorflow/tensorflow:1.15.2-gpu bash

# Check GPU
# Open python
# import tensorflow as tf
# tf.test.is_gpu_available()