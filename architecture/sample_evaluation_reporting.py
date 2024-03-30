import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Evaluate the model on the validation set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.view(inputs.size(0), -1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
# Calculate and print classification report
print("Classification Report:")
print(classification_report(all_labels, all_predictions))
# Calculate and print accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print("Accuracy:", accuracy)
# Calculate and print confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)
