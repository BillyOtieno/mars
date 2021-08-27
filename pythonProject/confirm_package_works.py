import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def main():
    print("TF Version: ", tf.__version__)
    print("TF-Hub Version: ", hub.__version__)
    print("Eager Mode Enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())

if __name__=="__main__":
    main()