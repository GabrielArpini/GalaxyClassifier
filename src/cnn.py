from torch import nn
import torch.nn.functional as F
import torch

class NeuralNet(nn.Module):
    """ Convolutional Neural Network arquitecture. """
    def __init__(self):
        super().__init__()
        self.layer1 = self.convblock(3,32)
        self.layer2 = self.convblock(32,64)
        self.layer3 = self.convblock(64,128)
        self.layer4 = self.convblock(128,256)
        self.layer5 = self.convblock(256,512)
        
        self.symmetry_mlp = nn.Sequential(
            nn.Linear(1, 32),  
            nn.ReLU(),
            nn.Linear(32, 16),  
            nn.ReLU()
        )
    
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.15)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512 + 16, 256) # + 16 from mlp
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)
        
        # Apply kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)        
    def forward(self, x, symmetry):
        x = self.layer1(x) # -> (32,256,256)
        x = self.layer2(x) # -> (64,128,128)
        x = self.layer3(x) # -> (128,64,64)
        x = self.layer4(x) # -> (256,32,32)
        x = self.layer5(x) # -> (512,16,16)


        x = self.global_avg_pool(x) # -> (512, 1, 1)
        x = torch.flatten(x, 1) # -> (512)

        # Run symmetry MLP
        sym_features = self.symmetry_mlp(symmetry.unsqueeze(1))
        x = torch.cat((x, sym_features), dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def convblock(self, in_channels,out_channels,kernel_size=3,padding=1,pool_size=2):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size)
        )


