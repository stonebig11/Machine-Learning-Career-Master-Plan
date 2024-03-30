import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Define the base model
base_model = SimpleModel(input_size=input_size, output_size=output_size)

# Create a bagging classifier with base models
ensemble_classifier = BaggingClassifier(
    base_model,
    n_estimators=num_models,
    base_estimator=DecisionTreeClassifier(),  # You can choose a different base estimator
    n_jobs=-1  # Use all available CPUs for parallel training
)
# predict with the ensemble framework
ensemble_predictions = ensemble_classifier.predict(val_inputs)
