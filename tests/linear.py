import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        # member variable
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.randn(out_features,in_features,device=device, dtype=dtype))
        
        # init weight
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=0.01)
    
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i, oi -> ...o', x, self.weights)
    
    