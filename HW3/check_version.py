import tensorflow as tf
import sklearn

# Check tensorflow version and GPU availibility


def Check_version():
    print(tf.__version__)
    print(sklearn.__version__)
    print(tf.test.is_gpu_available())


Check_version()
