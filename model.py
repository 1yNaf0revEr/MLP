from layers import Linear, ReLU, Tanh

class MLP:
    def __init__(self, input_dim=784, hidden_dim1=256, hidden_dim2=128, num_classes=10, activation="relu"):
        if activation == "relu":
            act1 = ReLU()
            act2 = ReLU()
        elif activation == "tanh":
            act1 = Tanh()
            act2 = Tanh()
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")

        self.layers = [
            Linear(input_dim, hidden_dim1),
            act1,
            Linear(hidden_dim1, hidden_dim2),
            act2,
            Linear(hidden_dim2, num_classes)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def linear_layers(self):
        return [layer for layer in self.layers if isinstance(layer, Linear)]

    def state_dict(self):
        state = {}
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                state[f"W{idx}"] = layer.W.copy()
                state[f"b{idx}"] = layer.b.copy()
                idx += 1
        return state

    def load_state_dict(self, state):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W = state[f"W{idx}"].copy()
                layer.b = state[f"b{idx}"].copy()
                idx += 1