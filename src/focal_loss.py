import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, beta=0.999, gamma=2.0, reduce=True,samples_per_class=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduce = reduce
        self.device = torch.accelerator.current_accelerator().type
        self.samples_per_class = samples_per_class
        
        #Compute effective number of samples per class
        if self.samples_per_class is not None:
            samples = torch.tensor(samples_per_class,dtype=torch.float32, device=self.device)
            self.effective_num = (1-torch.pow(beta,samples)) / (1-beta + 1e-8) # Epsilon to avoid divison error
            inverse_en = 1 / self.effective_num
            self.class_weights = inverse_en / inverse_en.sum() * len(samples_per_class)
            self.class_weights = self.class_weights.to(self.device)
        else:
            self.class_weights = None
    def forward(self, inputs, targets):
        # Fix device error
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal loss
        prob = F.softmax(inputs, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1-pt)**self.gamma) * ce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            focal_loss = focal_loss * weights
            
        return focal_loss.mean() if self.reduce else focal_loss

