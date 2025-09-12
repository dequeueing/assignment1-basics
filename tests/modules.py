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
    
    
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        # init member variable
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings,embedding_dim,device=device, dtype=dtype))
        
        # init weight
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=0.01)
        
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        indices = torch.LongTensor(token_ids)
        return self.weight[indices]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.gain = torch.nn.Parameter(torch.randn(d_model,device=device,dtype=dtype))
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast to avoid overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # perform RMSNorm
        square_divided = torch.einsum("...d,...d->...", x, x) / self.d_model
        rms_a = torch.sqrt(square_divided + self.eps)
        
        
        result = torch.einsum("...d,...->...d", x, 1/rms_a)
        result = torch.einsum("...d,d->...d", result, self.gain)
        
        # downcast to get back
        return result.to(in_dtype)