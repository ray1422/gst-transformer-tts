import json
from abc import ABC

import tensorflow as tf

from dataset import get_pinyin_dataset
from modules.transformer import *

""" hyper parm """
num_layers = 4
d_model = 128
dff = 513
num_heads = 8

input_size = 53
target_size = 80
dropout_rate = 0.1

epochs = 10
""" end hyper parm """


def main():
    model = Transformer(num_layers, d_model, num_heads, dff,
                        input_size, output_size=target_size,
                        pe_input=input_size,
                        pe_target=target_size,
                        rate=dropout_rate)

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule, ABC):
        def __init__(self, _d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = _d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def export(prenet_output, stops, post_output):
        pass

    def run_inference():
        with open("sentences.txt") as f:
            sentences = [x.strip().split(" ") for x in f.readlines()]
        with open("Token_Index_Dict.CHEWING.json") as f:
            token_dict = json.load(f)
        for sentence in sentences:
            _tokens = [token_dict[x] for x in sentence[0]]
            _tokens = [token_dict['<S>']] + tokens + [token_dict['<E>']]
            prenet_output, stops, post_output, attention_weights = model(_tokens)
            export(prenet_output, stops, post_output)

    def loss_function(_prenet_outputs, _stops, _postnet_outputs, _spe_length, _mels, _spes):
        prenet_loss = tf.reduce_mean(tf.square(_prenet_outputs - _mels), axis=[-1])
        postnet_loss = tf.reduce_mean(tf.square(_postnet_outputs - _spes), axis=[-1])

        _stops = _stops[:, :, 0]
        stop_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_spe_length, logits=_stops)
        stop_loss = tf.reduce_mean(stop_loss)
        alpha, beta, gamma = 1., 1., 1.
        losses = prenet_loss + postnet_loss
        losses *= tf.sequence_mask(_spe_length, maxlen=losses.shape[1], dtype=tf.float32)
        losses = tf.reduce_mean(tf.reduce_sum(losses, axis=-1) / tf.cast(_spe_length, dtype=tf.float32))
        loss = losses + stop_loss

        return loss, losses, stop_loss

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 513), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ]

    def test_step(inputs, mels, spes, spe_lengths):
        mels = tf.concat([tf.ones_like(mels[:, 0:1, :]), mels], axis=1)
        mel_inp = mels[:, :-1, ...]
        mel_real = mels[:, 1:, ...]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, mel_inp)

        prenet_output, stops, post_output, attention_weights = model(inputs, mel_inp,
                                                                     True,
                                                                     enc_padding_mask,
                                                                     combined_mask,
                                                                     dec_padding_mask, training=False)
        loss, losses, stop_loss = loss_function(prenet_output, stops, post_output, spe_lengths, mel_real, spes)
        return loss, losses, stop_loss

    # @tf.function(input_signature=train_step_signature)
    def train_step(inputs, mels, spes, spe_lengths):

        mels = tf.concat([tf.ones_like(mels[:, 0:1, :]), mels], axis=1)

        mel_inp = mels[:, :-1, ...]
        mel_real = mels[:, 1:, ...]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, mel_inp)

        with tf.GradientTape() as tape:
            prenet_output, stops, post_output, attention_weights = model(inputs, mel_inp,
                                                                         True,
                                                                         enc_padding_mask,
                                                                         combined_mask,
                                                                         dec_padding_mask)

            loss, losses, stop_loss = loss_function(prenet_output, stops, post_output, spe_lengths, mel_real, spes)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, losses, stop_loss

    train_dataset = get_pinyin_dataset("/work/ray200289/gst_tacotron_bert/TFRecords/train_chewing_bert_slim.tfrecords", 8, shuffle_size=30)
    valid_dataset = get_pinyin_dataset("/work/ray200289/gst_tacotron_bert/TFRecords/dev_chewing_bert_slim.tfrecords", 8)

    for epoch in range(epochs):
        for step, (mels, spectrograms, spectrogram_length, tokens, token_length) in enumerate(train_dataset):
            # print("token", tokens.shape)
            # print("mel", mels.shape)

            loss, losses, stop_loss = train_step(tokens, mels, spectrograms, spectrogram_length)
            print(f"steps {step}  loss: {loss:.5f}, losses: {losses:.5f}, stops: {stop_loss:.5f}")
            if step % 1000 == 0:
                run_inference()

        val_losses = []
        for step, (mels, spectrograms, spectrogram_length, tokens, token_length) in enumerate(valid_dataset):
            loss = test_step(tokens, mels, spectrograms, spectrogram_length)
            val_losses.append(loss)
            print("validating.. step {step} \r", end="")
        print(f"\n val_loss: {np.mean(val_losses): .5f}")


if __name__ == '__main__':
    main()
