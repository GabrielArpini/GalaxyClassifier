




import torch
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn


class NeuralNet(torch.nn.Module):
    """Equivariant CNN """
    
    def __init__(self, n_classes=10):
        super(NeuralNet, self).__init__()

        # Group action 
        self.r2_act = gspaces.rot2dOnR2(N=4)  # 4 folds 

        # Input type: RGB image (3 channels, trivial representation)
        in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        self.input_type = in_type
        
        # mixed representations
        out_type = nn.FieldType(self.r2_act, 
                               4*[self.r2_act.trivial_repr] + 
                               4*[self.r2_act.regular_repr])  
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 256, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),
            nn.FieldDropout(out_type, p=0.1)
        )

        
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 
                               8*[self.r2_act.trivial_repr] + 
                               8*[self.r2_act.regular_repr])  # 16 total
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),
            nn.FieldDropout(out_type, p=0.1)
        )
        
        # Early pooling 
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r2_act, 
                               12*[self.r2_act.trivial_repr] + 
                               12*[self.r2_act.regular_repr])  # 24 
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),
            nn.FieldDropout(out_type, p=0.15)
        )

    
        in_type = self.block3.out_type
        out_type = nn.FieldType(self.r2_act, 
                               16*[self.r2_act.trivial_repr] + 
                               16*[self.r2_act.regular_repr])  # 32 
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),
            nn.FieldDropout(out_type, p=0.15)
        )
        
        # Second pooling
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        
        in_type = self.block4.out_type
        out_type = nn.FieldType(self.r2_act, 
                               20*[self.r2_act.trivial_repr] + 
                               20*[self.r2_act.regular_repr])  # 40 total
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),
            nn.FieldDropout(out_type, p=0.2)
        )
        
        # spatial pooling
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=1)
        
        # Group pooling to make features rotationally invariant
        self.gpool = nn.GroupPooling(out_type)
        
        # Global spatial pooling
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # symmetry MLP
        self.symmetry_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 16),  # Reduced size
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(16, 8)   # Smaller output
        )

        # Calculate feature size after group pooling
        c = self.gpool.out_type.size

        # fully connected layers
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c + 8, 128),  # Reduced from 256
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.3),
            
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Linear(64, n_classes)
        )

    def forward(self, input: torch.Tensor, symmetry_input: torch.Tensor):
        # Wrap input in GeometricTensor
        x = nn.GeometricTensor(input, self.input_type)
        
        # Apply equivariant blocks 
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)  # Early pooling to reduce memory
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)  # Another pooling
        
        x = self.block5(x)
        x = self.pool3(x)

        # Pool over the group (make rotationally invariant)
        x = self.gpool(x)

        # Extract PyTorch tensor and apply global pooling
        x = x.tensor
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)

        # Process symmetry features
        sym_features = self.symmetry_mlp(symmetry_input.unsqueeze(1))
        
        # Combine features
        combined_features = torch.cat((x, sym_features), dim=1)

        # Final classification
        output = self.fully_net(combined_features)
        
        return output





# Normal neural net 

class NeuralNet3(torch.nn.Module):
    """ Convolutional Neural Network arquitecture. """
    def __init__(self):
        super().__init__()
        self.layer1 = self.convblock(3,32)
        self.layer2 = self.convblock(32,64)
        self.layer3 = self.convblock(64,128)
        self.layer4 = self.convblock(128,256)
        self.layer5 = self.convblock(256,512)
        
        self.symmetry_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 32),  
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),  
            torch.nn.ReLU()
        )
    
        self.dropout = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.15)
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = torch.nn.Linear(512 + 16, 256) # + 16 from mlp
        self.fc2 = torch.nn.Linear(256,128)
        self.fc3 = torch.nn.Linear(128,10)
        
        # Apply kaiming init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)        
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
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=pool_size)
        )


