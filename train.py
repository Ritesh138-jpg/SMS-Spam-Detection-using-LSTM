"""Train and evaluate an LSTM SMS spam classifier."""
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
DATA_PATH = Path("data/SMSSpamCollection")
MODEL_DIR = Path("models")
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 64


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download UCI SMS Spam Collection "
            "and put the extracted SMSSpamCollection file there."
        )
    df = pd.read_csv(
        path, sep="\t", header=None, names=["label", "message"], encoding="utf-8"
    )
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype("int32")
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0].drop_duplicates(subset=["message"])
    if df.empty:
        raise ValueError("Dataset is empty after validation.")
    return df


def build_model(vocab_size: int) -> tf.keras.Model:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                  input_length=MAX_LEN, mask_zero=True),
        LSTM(64, dropout=0.2),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main() -> None:
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    texts, labels = df["message"].values, df["label"].values

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    def encode(items):
        sequences = tokenizer.texts_to_sequences(items)
        return pad_sequences(
            sequences, maxlen=MAX_LEN, padding="post", truncating="post"
        )

    x_train_seq, x_val_seq, x_test_seq = map(
        encode, (x_train, x_val, x_test)
    )
    model = build_model(VOCAB_SIZE)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    model.fit(
        x_train_seq, y_train,
        validation_data=(x_val_seq, y_val),
        epochs=15, batch_size=32, callbacks=callbacks, verbose=1,
    )

    probabilities = model.predict(x_test_seq, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype("int32")
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_samples": int(len(y_test)),
    }

    model.save(MODEL_DIR / "sms_lstm.keras")
    joblib.dump(tokenizer, MODEL_DIR / "tokenizer.joblib")
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, target_names=["ham", "spam"], zero_division=0
    ))


if __name__ == "__main__":
    main()
