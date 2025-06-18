from torch import nn
import torch.functional as F
import torch

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=300, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=300, out_channels=200, kernel_size=3,padding=1,stride=1)
        self.fc1 = nn.Linear(200 * 64 * 64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x)) # -> (300, 256, 256)
        x = self.pool(x) # -> (300, 128, 128)
        x = F.relu(self.conv2(x)) # -> (200, 128, 128)
        x = self.pool(x) # -> (200, 64, 64)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        return x