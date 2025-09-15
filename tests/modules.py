import torch
from einops import rearrange

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
    
    
class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid = torch.nn.functional.sigmoid(x)
        return torch.einsum("...d,...d->...d", sigmoid, x)
    
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        """SwiGLU feed-forward network, comprising of a SiLU activation function and a GLU.

        Args:
            d_model (int): dimension of input
            d_ff (int): dimension of weight
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.silu = SiLU()
        
        self.w1 = torch.nn.Parameter(torch.randn(d_ff,d_model,device=device,dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.randn(d_model,d_ff,device=device,dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.randn(d_ff,d_model,device=device,dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = torch.einsum("fm,...m->...f", self.w1,x)
        silu_w1z = self.silu(w1x)
        w3x = torch.einsum("fm,...m->...f", self.w3,x)
        elementwise = torch.einsum("...f,...f->...f", silu_w1z, w3x)
        return torch.einsum("mf,...f->...m",self.w2, elementwise)
        
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(0, max_seq_len, device=device).float()  # shape: (max_seq_len,)
        
        # Create frequency indices: [0, 1, 2, ..., d_k//2-1] (0-indexed for easier computation)
        k_indices = torch.arange(0, d_k//2, device=device).float()  # shape: (d_k//2,)
        
        # Compute frequencies: theta^(-2k/d_k) for k in [0, 1, ..., d_k//2-1]
        freqs = self.theta ** (-2 * k_indices / d_k)  # shape: (d_k//2,)
        
        # Compute position * frequency for all position-frequency pairs using einsum
        pos_freqs = torch.einsum("p,k->pk", positions, freqs)
        
        # Precompute cos and sin values
        self.register_buffer('cos_cached', torch.cos(pos_freqs))  # shape: (max_seq_len, d_k//2)
        self.register_buffer('sin_cached', torch.sin(pos_freqs))  # shape: (max_seq_len, d_k//2)
    
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor):
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape but applied 
            positional information. Note that x may have arbitraty batch dimensions. Token positions are a tensor of shape
            (..., seq_len) specifying the token positions of x along the sequence dimension. 
            Should use token positions to slice the precomputed cosine and sine tensors along the sequence dimension.
            

        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): specify the token positions of x along the sequence dimension.
        """
                
        # Extract cos/sin values for the given token positions
        # token_positions shape: (..., seq_len)
        # We need to index into our cached cos/sin tensors
        cos_vals = self.cos_cached[token_positions]  # shape: (..., seq_len, d_k//2)
        sin_vals = self.sin_cached[token_positions]  # shape: (..., seq_len, d_k//2)
        
        # Reshape x to separate even/odd dimensions for rotation using rearrange
        # x shape: (..., seq_len, d_k) -> (..., seq_len, d_k//2, 2)
        x_pairs = rearrange(x, '... seq_len (pairs two) -> ... seq_len pairs two', two=2)
        
        # Split into even and odd components
        x_even = x_pairs[..., 0]  # shape: (..., seq_len, d_k//2)
        x_odd = x_pairs[..., 1]   # shape: (..., seq_len, d_k//2)
        
        # Apply rotation using element-wise operations
        # For each pair (x_even, x_odd), apply 2D rotation:
        # [cos -sin] [x_even]   [x_even * cos - x_odd * sin]
        # [sin  cos] [x_odd ] = [x_even * sin + x_odd * cos]
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
                
        # Recombine
        rotated_x = rearrange([rotated_even, rotated_odd], 'two ... seq_len pairs -> ... seq_len (pairs two)')
        
        return rotated_x