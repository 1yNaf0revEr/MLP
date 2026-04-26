import os
import json
import itertools
from model import MLP
from train import train_model

def hyperparameter_search(
    X_train,
    y_train,
    X_val,
    y_val,
    iterate_minibatches_fn,
    search_space,
    epochs=8,
    batch_size=128,
    save_dir="search_results"
):
    os.makedirs(save_dir, exist_ok=True)

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    results = []
    best_result = None

    for idx, combo in enumerate(itertools.product(*values), start=1):
        config = dict(zip(keys, combo))
        print(f"\n[Search {idx}] {config}")

        model = MLP(
            input_dim=784,
            hidden_dim1=config["hidden_dim1"],
            hidden_dim2=config["hidden_dim2"],
            num_classes=10,
            activation=config["activation"]
        )

        save_path = os.path.join(save_dir, f"best_model_{idx}.npz")

        history, best_val_acc = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            iterate_minibatches_fn=iterate_minibatches_fn,
            epochs=epochs,
            batch_size=batch_size,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            lr_decay=config["lr_decay"],
            save_path=save_path,
            verbose=False
        )

        result = {
            "config": config,
            "best_val_acc": float(best_val_acc),
            "model_path": save_path
        }
        results.append(result)

        print(f"best_val_acc = {best_val_acc:.4f}")

        if best_result is None or best_val_acc > best_result["best_val_acc"]:
            best_result = result

    with open(os.path.join(save_dir, "search_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return best_result, results