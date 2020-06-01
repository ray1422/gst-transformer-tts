from modules.transformer import Transformer
import tensorflow as tf

if __name__ == '__main__':

    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_size=8500, output_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)