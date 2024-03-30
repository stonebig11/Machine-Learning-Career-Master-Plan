import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
    def forward(self, predicted, target):
        # Custom loss calculation
        loss = torch.mean((predicted - target) ** 2)
        return loss
# Example usage
# Assuming some predicted and target tensors
predicted = torch.randn((5, 1), requires_grad=True)
target = torch.randn((5, 1), requires_grad=False)
# Create an instance of the custom loss function
custom_loss = CustomMSELoss()
# Calculate the loss
loss = custom_loss(predicted, target)
print("Custom MSE Loss:", loss.item())
