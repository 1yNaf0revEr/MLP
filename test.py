from utils import load_model, evaluate, predict, confusion_matrix

def test_best_model(model, model_path, X_test, y_test):
    load_model(model, model_path)
    test_loss, test_acc = evaluate(model, X_test, y_test)
    y_pred = predict(model, X_test)
    cm = confusion_matrix(y_test, y_pred, num_classes=10)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return test_loss, test_acc, cm