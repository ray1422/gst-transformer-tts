import numpy as np
import tensorflow as tf

MAX_MEL_LENGTH = 600
MAX_TOKEN_LENGTH = 50


def get_dataset():
    mels = tf.data.Dataset.from_tensor_slices(tf.zeros((64, 100, 80), dtype=tf.float32))
    spectrograms = tf.data.Dataset.from_tensor_slices(tf.zeros((64, 100, 512), dtype=tf.float32))
    tokens = tf.data.Dataset.from_tensor_slices(tf.ones((64, 8), dtype=tf.int32) + 1)
    spe_lengths = tf.data.Dataset.from_tensor_slices(tf.random.uniform((64,), maxval=20, dtype=tf.int32) + 1)
    ds = tf.data.Dataset.zip((tokens, mels, spectrograms, spe_lengths)).padded_batch(2, padded_shapes=(
        (MAX_TOKEN_LENGTH,),
        (MAX_MEL_LENGTH, 80),
        (MAX_MEL_LENGTH, 512),
        ()
    ))
    return ds
