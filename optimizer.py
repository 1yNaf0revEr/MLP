class SGD:
    def __init__(self, layers, lr=0.01, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * (layer.dW + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.db