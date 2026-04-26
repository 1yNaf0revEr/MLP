import os
import numpy as np
import matplotlib.pyplot as plt
from loss import cross_entropy_value

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def set_seed(seed=42):
    np.random.seed(seed)

def predict(model, X, batch_size=512):
    preds = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        logits = model.forward(xb)
        pred = np.argmax(logits, axis=1)
        preds.append(pred)
    return np.concatenate(preds)

def evaluate(model, X, y, batch_size=512):
    total_loss = 0.0
    total_correct = 0
    total = 0

    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model.forward(xb)
        loss = cross_entropy_value(logits, yb)

        total_loss += loss * len(xb)
        total_correct += np.sum(np.argmax(logits, axis=1) == yb)
        total += len(xb)

    return total_loss / total, total_correct / total

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def save_model(model, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(path, **model.state_dict())

def load_model(model, path):
    data = np.load(path)
    state = {k: data[k] for k in data.files}
    model.load_state_dict(state)

def plot_history(history, save_path="training_curves.png"):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Decay")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_first_layer_weights(model, num_units=16, save_path="first_layer_weights.png"):
    W = model.linear_layers()[0].W
    num_units = min(num_units, W.shape[1])
    cols = 4
    rows = int(np.ceil(num_units / cols))

    plt.figure(figsize=(cols * 2.8, rows * 2.8))
    for i in range(num_units):
        plt.subplot(rows, cols, i + 1)
        weight_img = W[:, i].reshape(28, 28)
        vmax = np.max(np.abs(weight_img))
        plt.imshow(weight_img, cmap="seismic", vmin=-vmax, vmax=vmax)
        plt.axis("off")
        plt.title(f"Unit {i}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def show_misclassified_examples(model, X, y, class_names=CLASS_NAMES, num_examples=12, save_path="misclassified.png"):
    y_pred = predict(model, X)
    wrong_idx = np.where(y_pred != y)[0]
    num_examples = min(num_examples, len(wrong_idx))

    plt.figure(figsize=(12, 8))
    for i in range(num_examples):
        idx = wrong_idx[i]
        plt.subplot(3, 4, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{class_names[y[idx]]}\nP:{class_names[y_pred[idx]]}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()