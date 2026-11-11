import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from seqeval.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification

HERE = Path(__file__).parent
OUT = HERE / "workflow"
MAX_LEN = 512


class NER:
    """Resume NER: clean the dataturks data, tag words, fine-tune DistilBERT, plot label counts."""

    def __init__(self):
        OUT.mkdir(exist_ok=True)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(HERE / "tokenizer"))

    def load_data(self):
        """Convert ner.json to (text, entities) with whitespace-trimmed spans."""
        self.df = pd.read_json(HERE / "ner.json", lines=True).drop(['extras'], axis=1)
        self.df['content'] = self.df['content'].str.replace("\n", " ")

        data = []
        with open(HERE / "ner.json", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item['content'].replace("\n", " ")
                entities = []
                for ann in item['annotation'] or []:
                    point = ann['points'][0]
                    labels = ann['label'] if isinstance(ann['label'], list) else [ann['label']]
                    start, end, ptext = point['start'], point['end'], point['text']
                    start += len(ptext) - len(ptext.lstrip())
                    end -= len(ptext) - len(ptext.rstrip())
                    for label in labels:
                        entities.append([start, end + 1, label])
                data.append((text, entities))

        for text, entities in data:
            for ent in entities:
                while ent[0] < len(text) and re.match(r'\s', text[ent[0]]):
                    ent[0] += 1
                while ent[1] > 1 and re.match(r'\s', text[ent[1] - 1]):
                    ent[1] -= 1
        self.data = data

    def tag_words(self):
        """One tag per whitespace separated word, 'Empty' when there is no entity."""
        rows = []
        for text, ents in self.data:
            tags = ["Empty"] * len(text.split())
            n = 0
            start = 0
            last_space = text.rfind(' ')
            for i in range(len(text)):
                if text[i] == " " and text[i + 1] != " ":
                    for ent in reversed(ents):
                        if start >= ent[0] and i <= ent[1]:
                            tags[n] = ent[2]
                            break
                    start = i + 1
                    n += 1
                if i == last_space:
                    for ent in reversed(ents):
                        if last_space >= ent[0] and len(text) <= ent[1]:
                            tags[n] = ent[2]
                            n += 1
            rows.append(tags)

        self.unique_tags = sorted({t for row in rows for t in row})
        self.tag2id = {t: i for i, t in enumerate(self.unique_tags)}
        self.id2tag = {i: t for t, i in self.tag2id.items()}
        self.tags = pad_sequences([[self.tag2id[t] for t in row] for row in rows],
                                  maxlen=MAX_LEN, value=self.tag2id["Empty"],
                                  padding="post", dtype="long", truncating="post")

    def tokenize(self):
        enc = self.tokenizer(self.df['content'].values.tolist(), truncation=True,
                             padding='max_length', max_length=MAX_LEN)
        labels = []
        for i, tag_row in enumerate(self.tags):
            # special tokens get -100 so the loss ignores them, every sub-token gets its word's tag
            labels.append([-100 if w is None else tag_row[w] for w in enc.word_ids(batch_index=i)])
        enc["labels"] = labels
        self.enc = enc
        self.dataset = tf.data.Dataset.from_tensor_slices((enc['input_ids'], enc['labels']))

    def train(self, epochs=10):
        self.model = TFDistilBertForTokenClassification.from_pretrained(str(HERE / "model"), num_labels=len(self.unique_tags))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                           loss=self.model.hf_compute_loss, metrics=['accuracy'])
        self.model.fit(self.dataset.batch(4), epochs=epochs, batch_size=4)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
        return np.argmax(self.model(inputs).logits, axis=2)

    def evaluate(self):
        n = len(self.enc['labels'])
        true_labels = [[self.id2tag.get(t, "Empty") for t in row] for row in self.enc['labels']]
        logits = self.model.predict(self.dataset)['logits'].reshape(n, -1, len(self.unique_tags))
        preds = np.argmax(logits, axis=-1)
        pred_labels = [[self.id2tag.get(t, "Empty") for t in row] for row in preds]

        print(Counter(np.array(true_labels).flatten()))
        print(classification_report(true_labels, pred_labels))

        for labels, name in [(true_labels, "true_label_hist"), (pred_labels, "predicted_label_hist")]:
            plt.figure()
            plt.hist(np.array(labels).flatten())
            plt.xticks(rotation='vertical')
            plt.savefig(OUT / f"{name}.png", bbox_inches="tight")
            plt.close()

    def run(self):
        self.load_data()
        self.tag_words()
        self.tokenize()
        self.train()
        print(self.predict("Manisha Bharti. 3.5 years of professional IT experience in Banking and Finance domain"))
        self.evaluate()


if __name__ == "__main__":
    NER().run()

