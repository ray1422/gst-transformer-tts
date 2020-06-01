import tensorflow as tf
import numpy as np
import tensorflow as tf

MAX_MEL_LENGTH = 600
MAX_TOKEN_LENGTH = 50


def get_dataset():
    mels = tf.data.Dataset.from_tensor_slices(tf.random.uniform((64, 100, 80), dtype=tf.float32))
    spectrograms = tf.data.Dataset.from_tensor_slices(tf.random.uniform((64, 100, 512), dtype=tf.float32))
    tokens = tf.data.Dataset.from_tensor_slices(tf.random.uniform((256, 8), maxval=20, dtype=tf.int32))
    ds = tf.data.Dataset.zip((mels, spectrograms, tokens)).padded_batch(1, padded_shapes=(
        (MAX_MEL_LENGTH, 80),
        (MAX_MEL_LENGTH, 512),
        (MAX_TOKEN_LENGTH,)
    ))
    return ds
