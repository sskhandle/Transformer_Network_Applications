from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

HERE = Path(__file__).parent
OUT = HERE / "workflow"

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 64

TEXTS = ['king queen man woman dog wolf football basketball red green yellow',
         'man queen yellow basketball green dog  woman football  king red wolf']


class PreprocessingDemo:
    """Positional encoding checks plus GloVe embedding + positional encoding plots."""

    def __init__(self):
        OUT.mkdir(exist_ok=True)
        self.pos_encoding = self.positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

    @staticmethod
    def positional_encoding(positions, d):
        angle_rads = np.arange(positions)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d)[np.newaxis, :] // 2)) / np.float32(d))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def _save_heatmap(self, matrix, xlabel, name):
        plt.figure()
        plt.pcolormesh(matrix, cmap='RdBu')
        plt.xlabel(xlabel)
        plt.xlim((0, matrix.shape[1]))
        plt.ylabel('Position')
        plt.colorbar()
        plt.savefig(OUT / f"{name}.png", bbox_inches="tight")
        plt.close()

    def positional_encoding_plots(self):
        pe = self.pos_encoding
        self._save_heatmap(pe[0], 'd', "positional_encoding_heatmap")
        print("norm of pos 34:", float(tf.norm(pe[0, 34, :])))
        print("norm of diff pos 70 vs 72:", float(tf.norm(pe[0, 70, :] - pe[0, 72, :])))

        corr = tf.matmul(pe, pe, transpose_b=True).numpy()[0]
        self._save_heatmap(corr, 'Position', "correlation_heatmap")

        eu = np.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
        for a in range(MAX_SEQUENCE_LENGTH):
            for b in range(a + 1, MAX_SEQUENCE_LENGTH):
                eu[a, b] = eu[b, a] = tf.norm(pe[0, a] - pe[0, b])
        self._save_heatmap(eu, 'Position', "euclidean_distance_heatmap")

    def load_embeddings(self):
        # needs glove.6B.100d.txt in src/preprocessing/glove/
        index = {}
        with open(HERE / "glove" / "glove.6B.100d.txt", encoding="utf8") as f:
            for line in f:
                values = line.split()
                index[values[0]] = np.asarray(values[1:], dtype='float32')
        print("Found %s word vectors." % len(index))

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(TEXTS)
        self.sequences = tokenizer.texts_to_sequences(TEXTS)
        self.word_index = tokenizer.word_index
        data = pad_sequences(self.sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

        matrix = np.zeros((len(self.word_index) + 1, EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if word in index:
                matrix[i] = index[word]  # words missing from glove stay all-zeros

        layer = Embedding(len(self.word_index) + 1, EMBEDDING_DIM,
                          embeddings_initializer=tf.keras.initializers.Constant(matrix),
                          trainable=False)
        self.embedding = layer(data)
        print("embedding shape:", self.embedding.shape)

    def plot_words(self, embedding, sentence, name):
        seq = self.sequences[sentence]
        pca = PCA(n_components=2)
        xy = pca.fit_transform(embedding[sentence, 0:len(seq), :])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(xy[:, 0], xy[:, 1])
        words = list(self.word_index.keys())
        for i, index in enumerate(seq):
            ax.annotate(words[index - 1], (xy[i, 0], xy[i, 1]))
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    def embedding_plots(self):
        pe = self.pos_encoding
        self.plot_words(self.embedding, 0, "words_sentence1")
        self.plot_words(self.embedding, 1, "words_sentence2")

        combined = self.embedding * 1.0 + pe * 1.0
        self.plot_words(combined, 0, "words_pos_sentence1")
        self.plot_words(combined, 1, "words_pos_sentence2")

        # heavier weight on the positional part makes word order visible
        combined = self.embedding * 1 + pe * 10
        self.plot_words(combined, 0, "words_pos_w10_sentence1")
        self.plot_words(combined, 1, "words_pos_w10_sentence2")

    def run(self):
        self.positional_encoding_plots()
        self.load_embeddings()
        self.embedding_plots()


if __name__ == "__main__":
    PreprocessingDemo().run()

