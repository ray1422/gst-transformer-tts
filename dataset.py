import json

import tensorflow as tf
from tqdm import tqdm


MAX_TOKEN_LENGTH = 72
MAX_SPE_LENGTH = 600

def get_pinyin_dataset(filename, batch_size=32, shuffle_size=1):
    feature = {
        'mels': tf.io.VarLenFeature(dtype=tf.float32),
        'spectrograms': tf.io.VarLenFeature(dtype=tf.float32),
        'spectrogram_length': tf.io.FixedLenFeature([], dtype=tf.int64),
        'tokens': tf.io.VarLenFeature(dtype=tf.int64),
        'token_length': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        feature_dict = tf.io.parse_single_example(example_proto, feature)
        mels = tf.sparse.to_dense(feature_dict['mels'])
        spectrograms = tf.sparse.to_dense(feature_dict['spectrograms'])
        spectrogram_length = feature_dict['spectrogram_length']
        tokens = tf.sparse.to_dense(feature_dict['tokens'])
        token_length = feature_dict['token_length']
        mels = tf.reshape(mels, shape=(spectrogram_length, -1))[:MAX_SPE_LENGTH, :]
        spectrograms = tf.reshape(spectrograms, shape=(spectrogram_length, -1))[:MAX_SPE_LENGTH, :]
        spectrogram_length = tf.clip_by_value(spectrogram_length, 0, MAX_SPE_LENGTH - 1)
        return mels, spectrograms, spectrogram_length, tokens, token_length

    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames).shuffle(shuffle_size)
    dataset = raw_dataset.map(_parse_function)
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padding_values=(0.,
                                                   0.,
                                                   tf.constant(0, dtype=tf.int64),
                                                   tf.constant(0, dtype=tf.int64),
                                                   tf.constant(0, dtype=tf.int64)),
                                   padded_shapes=((MAX_SPE_LENGTH, 80),
                                                  (MAX_SPE_LENGTH, 513),
                                                  (),
                                                  MAX_TOKEN_LENGTH,
                                                  ()))

    return dataset
