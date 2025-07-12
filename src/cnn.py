from torch import nn
import torch.nn.functional as F
import torch

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128,256, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(256,512, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(512,1024, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.50)
        self.fc1 = nn.Linear(1024 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,10)
        
        # Apply kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> (32, 256, 256)
        x = self.pool(x)  # -> (32, 128, 128)
        x = F.relu(self.conv2(x))  # -> (64, 128, 128)
        x = self.pool(x)  # -> (64, 64, 64)
        x = F.relu(self.conv3(x))  # -> (128, 64, 64)
        x = self.pool(x)  # -> (128, 32, 32)
        x = F.relu(self.conv4(x)) # -> (256, 32, 32)
        x = self.pool(x) # -> (256, 16, 16)
        x = F.relu(self.conv5(x)) # -> (512, 16, 16)
        x = self.pool(x) # -> (512, 8, 8)
        x = F.relu(self.conv6(x)) # -> 1024, 8, 8)
        x = self.pool(x) # -> (1024, 4, 4)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        
        return x
