import os
import gzip
import struct
import urllib.request
import numpy as np

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def download_fashion_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for filename in FILES.values():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            url = f"{BASE_URL}/{filename}"
            urllib.request.urlretrieve(url, path)

def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    return data

def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return data

def load_fashion_mnist(data_dir="data", val_ratio=0.1, seed=42):
    download_fashion_mnist(data_dir)

    X_train = read_idx_images(os.path.join(data_dir, FILES["train_images"]))
    y_train = read_idx_labels(os.path.join(data_dir, FILES["train_labels"]))
    X_test = read_idx_images(os.path.join(data_dir, FILES["test_images"]))
    y_test = read_idx_labels(os.path.join(data_dir, FILES["test_labels"]))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train))
    val_size = int(len(X_train) * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(X, y, batch_size=128, shuffle=True):
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]