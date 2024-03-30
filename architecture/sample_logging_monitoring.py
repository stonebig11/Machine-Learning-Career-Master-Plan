import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Create a PyTorch dataset and data loader
def create_data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return train_loader
# Train a PyTorch model with logging
def train_model():
    input_size = 28 * 28  # MNIST image size
    output_size = 10  # Number of classes
    learning_rate = 0.001
    epochs = 5
    model = SimpleModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = create_data_loader()
    # Create a TensorBoard writer
    writer = SummaryWriter()
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input for the fully connected layer
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Log loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
    # Close the TensorBoard writer
    writer.close()
    print("Training complete.")

# Example usage
train_model()
