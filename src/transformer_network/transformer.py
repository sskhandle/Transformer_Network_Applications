from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Dropout, LayerNormalization

HERE = Path(__file__).parent
OUT = HERE / "workflow"


def get_angles(pos, k, d):
    return pos / np.power(10000, (2 * (k // 2)) / np.float32(d))


def positional_encoding(positions, d):
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)


def create_padding_mask(token_ids):
    seq = 1 - tf.cast(tf.math.equal(token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :]


def create_look_ahead_mask(sequence_length):
    return tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        logits += (1. - mask) * -1e9
    weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(weights, v), weights


def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),
        tf.keras.layers.Dense(embedding_dim),
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """Self-attention + feed forward, each with residual and layer norm."""

    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn = self.mha(x, x, x, mask)
        skip = self.layernorm1(x + attn)
        ffn_out = self.dropout_ffn(self.ffn(skip), training=training)
        return self.layernorm2(ffn_out + skip)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, fully_connected_dim, dropout_rate, layernorm_eps)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training, mask)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    """Masked self-attention, encoder-decoder attention, then feed forward."""

    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        out1, w1 = self.mha1(x, x, x, look_ahead_mask, return_attention_scores=True)
        q1 = self.layernorm1(out1 + x)
        out2, w2 = self.mha2(q1, enc_output, enc_output, padding_mask, return_attention_scores=True)
        out2 = self.layernorm2(out2 + q1)
        ffn_out = self.dropout_ffn(self.ffn(out2), training=training)
        return self.layernorm3(ffn_out + out2), w1, w2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)
        self.dec_layers = [DecoderLayer(embedding_dim, num_heads, fully_connected_dim, dropout_rate, layernorm_eps)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1_self_att'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2_decenc_att'] = block2
        return x, attention_weights


class Transformer(tf.keras.Model):
    """Full encoder-decoder transformer."""

    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                 target_vocab_size, max_positional_encoding_input, max_positional_encoding_target,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.encoder = Encoder(num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                               max_positional_encoding_input, dropout_rate, layernorm_eps)
        self.decoder = Decoder(num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
                               max_positional_encoding_target, dropout_rate, layernorm_eps)
        self.final_layer = Dense(target_vocab_size, activation='softmax')

    def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask)
        return self.final_layer(dec_output), attention_weights


class TransformerDemo:
    """Runs the unit tests and the small demos from the original notebook."""

    def run_tests(self):
        from public_tests import (get_angles_test, positional_encoding_test, scaled_dot_product_attention_test,
                                  EncoderLayer_test, DecoderLayer_test, Decoder_test, Transformer_test)
        get_angles_test(get_angles)
        positional_encoding_test(positional_encoding, get_angles)
        scaled_dot_product_attention_test(scaled_dot_product_attention)
        EncoderLayer_test(EncoderLayer)
        DecoderLayer_test(DecoderLayer, create_look_ahead_mask)
        Decoder_test(Decoder, create_look_ahead_mask, create_padding_mask)
        Transformer_test(Transformer, create_look_ahead_mask, create_padding_mask)

    def plot_positional_encoding(self):
        OUT.mkdir(exist_ok=True)
        pe = positional_encoding(50, 512)
        print(pe.shape)
        plt.pcolormesh(pe[0], cmap='RdBu')
        plt.xlabel('d')
        plt.xlim((0, 512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.savefig(OUT / "positional_encoding_heatmap.png", bbox_inches="tight")
        plt.close()

    def show_masks(self):
        x = tf.constant([[7., 6., 0., 0., 1.], [1., 2., 3., 0., 0.], [0., 0., 0., 4., 5.]])
        print(create_padding_mask(x))
        print(tf.keras.activations.softmax(x))
        print(tf.keras.activations.softmax(x + (1 - create_padding_mask(x)) * -1.0e9))
        print(create_look_ahead_mask(3))

    def run(self):
        self.plot_positional_encoding()
        self.show_masks()
        self.run_tests()


if __name__ == "__main__":
    TransformerDemo().run()

