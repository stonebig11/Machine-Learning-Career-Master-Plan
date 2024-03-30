import torch
import torch.nn as nn
class GeneralizedModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64], activations=[nn.ReLU()], dropout_p=0.0):
        super(GeneralizedModel, self).__init__()
        self.layers = []
        prev_size = input_size
        # Create hidden layers
        for size, activation in zip(hidden_sizes, activations):
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(activation)
            if dropout_p > 0:
                self.layers.append(nn.Dropout(p=dropout_p))
            prev_size = size
        # Output layer
        self.layers.append(nn.Linear(prev_size, output_size))
        # Assemble the layers into a Sequential module
        self.model = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.model(x)
# Example usage
input_size = 10
output_size = 1
hidden_sizes = [32, 16]
activations = [nn.ReLU(), nn.Tanh()]
dropout_p = 0.2
# Create an instance of the generalized model
model = GeneralizedModel(input_size, output_size, hidden_sizes, activations, dropout_p)
# Print the model architecture
print(model)
