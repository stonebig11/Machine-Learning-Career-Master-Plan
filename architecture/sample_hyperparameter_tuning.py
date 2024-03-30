import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize the model
model = SimpleModel(input_size=784, output_size=10)  # Adjust input_size and output_size accordingly
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Define hyperparameters to tune
param_grid = {'lr': [0.001, 0.01, 0.1], 'hidden_size': [32, 64, 128]}
# Define the training function
def train_model(lr, hidden_size, X_train, y_train, X_val, y_val):
	...
    return val_loss.item(), val_accuracy
# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
