import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0,beta=0.999, gamma=2.0, reduce=True,samples_per_class=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduce = reduce
        self.samples_per_class = samples_per_class
        
        #Compute effective number of samples per class
        if self.samples_per_class is not None:
            self.effective_num = (1-torch.pow(beta,torch.tensor(samples_per_class,dtype=torch.float32))) / (1-beta)
            inverse_en = 1 / self.effective_num
            self.class_weights = inverse_en / inverse_en.sum() * len(samples_per_class)
        else:
            self.class_weights = None
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1-pt)**self.gamma*ce_loss)
        return focal_loss.mean() if self.reduce else focal_loss

