import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,beta=0.999, gamma=2.0, reduce=True,samples_per_class=None,device='cpu'):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.beta = beta
        self.reduce = reduce
        self.device = device
        self.samples_per_class = samples_per_class
        
        #Compute effective number of samples per class
        if self.samples_per_class is not None:
            effective_num = (1-torch.pow(beta,torch.tensor(samples_per_class,dtype=torch.float32))) / (1-beta)
            self.inverse_en = (1 / effective_num).to(device)
            
    def forward(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1-pt)**self.gamma)*ce_loss
        cb_focal = self.inverse_en[targets] * focal_loss 
        
        return cb_focal.mean() if self.reduce else cb_focal
    
