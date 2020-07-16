docker run -it --rm --name tf2 --gpus "device=0" tensorflow/tensorflow:2.1.1-gpu bash

# Check GPU
# Open python
# import tensorflow as tf
# tf.test.is_gpu_available()

# docker run --rm --name notebook -p 8888:8888 tensorflow/tensorflow:latest-jupyter