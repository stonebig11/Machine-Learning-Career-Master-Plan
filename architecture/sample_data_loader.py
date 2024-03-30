import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Custom dataset class for loading data using PyTorch.
        Parameters:
        - file_path (str): Path to the data file.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = torch.from_numpy(np.loadtxt(file_path, delimiter=','))  # Assuming CSV data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
# Example usage
file_path = 'example_data.csv'
custom_dataset = CustomDataset(file_path)

# Create a DataLoader
batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
