import torch
import torch.nn as nn

class PoseEstimationCNN(nn.Module):
    def __init__(self, input_dim, num_joints=17):
        super(PoseEstimationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Dynamically compute the flattened size
        dummy_input = torch.zeros(1, 1, input_dim, 5)  # Match actual input dimensions
        flattened_size = self._get_flattened_size(dummy_input)
        print(f"Flattened size for fc1: {flattened_size}")  # Debugging

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_joints * 3)

    def _get_flattened_size(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        print(f"Shape after conv1 and conv2: {x.shape}")  # Debugging shapes
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 17, 3)


"""class PoseEstimationCNN(nn.Module):
    def __init__(self, input_dim, num_joints=17):
        super(PoseEstimationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_dim, 256)
        self.fc2 = nn.Linear(256, num_joints * 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 17, 3)"""
