from modules.transformer import Transformer, create_masks
import tensorflow as tf

if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_size=50, output_size=512,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((3, 62), maxval=20, dtype=tf.int32)  # token
    temp_target = tf.random.uniform((3, 90, 512))
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(temp_input, temp_target)

    prenet_output, stops, post_output, attention_weights = sample_transformer(temp_input, temp_target, training=True,
                                                                              enc_padding_mask=enc_padding_mask,
                                                                              look_ahead_mask=combined_mask,
                                                                              dec_padding_mask=dec_padding_mask)

    print(post_output.shape)  # (batch_size, tar_seq_len, target_vocab_size)
