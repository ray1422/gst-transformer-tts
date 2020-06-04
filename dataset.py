import json

import tensorflow as tf

with open('hyper_parameters.json', 'r') as f:
    hp_dict = json.load(f)


def get_pinyin_dataset(filename, batch_size=hp_dict["Train"]["Batch_Size"]):
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
        mels = tf.reshape(mels, shape=(spectrogram_length, -1))
        spectrograms = tf.reshape(spectrograms, shape=(spectrogram_length, -1))

        return mels, spectrograms, spectrogram_length, tokens, token_length

    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    dataset = raw_dataset.map(_parse_function)
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padding_values=(-1.,
                                                   -1.,
                                                   tf.constant(-1, dtype=tf.int64),
                                                   tf.constant(-1, dtype=tf.int64),
                                                   tf.constant(-1, dtype=tf.int64)),
                                   padded_shapes=((hp_dict["Sound"]["max_spe_length"], hp_dict["Sound"]["Mel_Dim"]),
                                                  (hp_dict["Sound"]["max_spe_length"], hp_dict["Sound"]["Spectrogram_Dim"]),
                                                  (),
                                                  hp_dict["Token"]["max_length"],
                                                  ()))

    return dataset
