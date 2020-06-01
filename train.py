from abc import ABC

import tensorflow as tf

from fake_dataset import get_dataset
from modules.transformer import *

""" hyper parm """
num_layers = 4
d_model = 128
dff = 512
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

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None, target_size), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, targets):
        def loss_function(outputs):
            # TODO
            return tf.reduce_mean(outputs - targets)

        tar_inp = targets[:, :-1, ...]
        tar_real = targets[:, 1:, ...]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(tar_inp, tar_inp)
        print(enc_padding_mask)
        print(combined_mask)
        print(dec_padding_mask)
        with tf.GradientTape() as tape:
            predictions, _ = model(inputs, tar_inp,
                                   True,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)
            loss = loss_function(tar_real)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for epoch in range(epochs):
        for mels, spectrograms, tokens in get_dataset():
            train_step(tokens, mels)


if __name__ == '__main__':
    main()
