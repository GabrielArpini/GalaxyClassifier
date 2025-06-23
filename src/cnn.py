from torch import nn
import torch.nn.functional as F
import torch

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128) 
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> (32, 256, 256)
        x = self.pool(x)  # -> (32, 128, 128)
        x = F.relu(self.conv2(x))  # -> (64, 128, 128)
        x = self.pool(x)  # -> (64, 64, 64)
        x = F.relu(self.conv3(x))  # -> (32, 64, 64)
        x = self.pool(x)  # -> (32, 32, 32)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
