from torch import nn

class MLP(nn.Module):
  def __init__(self,n_inputs,n_hidden1,n_hidden2,n_classes):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Flatten(),
      nn.Linear(n_inputs,n_hidden1),
      nn.ReLU(),
      nn.Linear(n_hidden1,n_hidden2),
      nn.ReLU(),
      nn.Linear(n_hidden2,n_classes)
    )
  def forward(self, X):
    return self.mlp(X)