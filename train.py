from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD
from utils import evaluate, save_model, load_model

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    iterate_minibatches_fn,
    epochs=20,
    batch_size=128,
    lr=0.05,
    weight_decay=1e-4,
    lr_decay=0.0,
    save_path="checkpoints/best_model.npz",
    verbose=True
):
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.linear_layers(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        current_lr = lr / (1.0 + lr_decay * (epoch - 1))
        optimizer.lr = current_lr

        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in iterate_minibatches_fn(X_train, y_train, batch_size=batch_size, shuffle=True):
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            grad = criterion.backward()
            model.backward(grad)
            optimizer.step()

            train_loss_sum += loss * len(xb)
            train_count += len(xb)

        train_loss = train_loss_sum / train_count
        val_loss, val_acc = evaluate(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, save_path)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"lr={current_lr:.5f} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

    load_model(model, save_path)
    return history, best_val_acc