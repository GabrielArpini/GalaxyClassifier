#from torch import nn
import torch.nn.functional as F
import torch
from escnn import gspaces
from escnn import nn

class NeuralNet(nn.Module):
    """Equivariant CNN implementation."""
    def __init__(self, num_classes=10):
        super(NeuralNet, self).__init__()

        # Create Group action
        r2_act = gspaces.rot2dOnR2(N=8) # 8 rotations

        # Create input type
        in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        self.input_type = in_type
        
        # Create conv1 
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.trivial_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # Create conv2 
        # Uses the old out_type as in_type for this
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.trivial_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        # Create first pooling layer

        self.pool1 = nn.SequentialModule(
            nn.PointWiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # Conv3
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.trivial_repr]) 
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # Conv4
        in_type = self.block3.out_type
        out_type = nn.FieldType(self.r2_act, 256*[self.r2_act.trivial_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # Conv5
        in_type = self.block4.out_type
        out_type = nn.FieldType(self.r2_act, 512*[self.r2_act.trivial_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # Conv6 
        in_type = self.block5.out_type
        out_type = nn.FieldType(self.r2_act, 256*[self.r2_act.trivial_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ) 
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(128, n_classes),
        )

        def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x
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


