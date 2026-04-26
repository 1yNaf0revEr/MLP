from data import load_fashion_mnist, iterate_minibatches
from model import MLP
from train import train_model
from search import hyperparameter_search
from test import test_best_model
from utils import (
    set_seed,
    plot_history,
    plot_first_layer_weights,
    plot_confusion_matrix,
    show_misclassified_examples
)

def main():
    set_seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist(
        data_dir="data",
        val_ratio=0.1,
        seed=42
    )

    search_space = {
        "hidden_dim1": [256, 512],
        "hidden_dim2": [128, 256],
        "activation": ["relu", "tanh"],
        "lr": [0.1, 0.05],
        "weight_decay": [1e-4, 5e-4],
        "lr_decay": [0.0, 0.05]
    }

    best_result, all_results = hyperparameter_search(
        X_train,
        y_train,
        X_val,
        y_val,
        iterate_minibatches_fn=iterate_minibatches,
        search_space=search_space,
        epochs=8,
        batch_size=128,
        save_dir="search_results"
    )

    print("\nBest config:")
    print(best_result)

    best_config = best_result["config"]

    model = MLP(
        input_dim=784,
        hidden_dim1=best_config["hidden_dim1"],
        hidden_dim2=best_config["hidden_dim2"],
        num_classes=10,
        activation=best_config["activation"]
    )

    history, best_val_acc = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        iterate_minibatches_fn=iterate_minibatches,
        epochs=20,
        batch_size=128,
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        lr_decay=best_config["lr_decay"],
        save_path="checkpoints/final_best_model.npz",
        verbose=True
    )

    plot_history(history, save_path="training_curves.png")
    plot_first_layer_weights(model, num_units=16, save_path="first_layer_weights.png")

    test_loss, test_acc, cm = test_best_model(
        model,
        "checkpoints/final_best_model.npz",
        X_test,
        y_test
    )

    plot_confusion_matrix(cm, save_path="confusion_matrix.png")
    show_misclassified_examples(model, X_test, y_test, num_examples=12, save_path="misclassified.png")

if __name__ == "__main__":
    main()