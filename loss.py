import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.probs = probs
        self.y_true = y_true

        n = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(n), y_true] + 1e-12))
        return loss

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y_true] -= 1
        grad /= n
        return grad

def cross_entropy_value(logits, y_true):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-12))
    return loss