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
    
    
class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x:torch.Tensor, dim:int):
        # get the maximum value for the targeted dimension
        maximum = torch.max(x, dim=dim, keepdim=True).values
        
        # minus the maximum value
        x = x - maximum
        
        # calculate exp for each element 
        x = torch.exp(x)
        
        # get the sum
        exp_sum = torch.sum(x, dim=dim, keepdim=True)
        
        return x / exp_sum
        
        
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
        
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor):
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        # Q @ K / sqrt(dk)
        dk = torch.tensor(Q.shape[-1])
        QK = torch.einsum("... q d, ... k d -> ... q k", Q, K) / torch.sqrt(dk)
        
        # add mask 
        if mask is not None:
            QK = QK.masked_fill(mask == False, float('-inf'))
        
        # softmax 
        sfm_QK = self.softmax(QK, dim=-1)
        
        # multiply V
        return torch.einsum("... a l, ... l v->...av", sfm_QK, V)

class MultiheadAttention(torch.nn.Module):
    def __init__(self,d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = torch.nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        self.k_proj = torch.nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        self.v_proj = torch.nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        self.o_proj = torch.nn.Parameter(torch.randn(d_model,d_model,device=device,dtype=dtype))
        
        self.attention = Attention()
        
        # Initialize RoPE if theta is provided
        if self.theta is not None:
            if max_seq_len is None:
                raise ValueError("max_seq_len must be provided when using RoPE (theta is not None)")
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        
        
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor=None): 
        # get sequence length from x 
        seq_len = x.shape[-2]
        
        # x shape: sequence_length, d_model
        # q/k/v shape: ...  d_model
        # q = torch.einsum("... d, d v->...v", x, self.q_proj)
        # k = torch.einsum("... d, d v->...v", x, self.k_proj)
        # v = torch.einsum("... d, d v->...v", x, self.v_proj)
        
        # fix: pay attention to the order of dimension, different!
        q = torch.einsum("... s d, v d -> ... s v", x, self.q_proj)
        k = torch.einsum("... s d, v d -> ... s v", x, self.k_proj)
        v = torch.einsum("... s d, v d -> ... s v", x, self.v_proj)
        
        # rearrange into multihead 
        # ...  d_model -> ... num_head, d_head        
        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)
        
        # apply rope if not none
        if self.theta:
            # Generate default token positions if not provided
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                # Add batch dimensions to match x's shape
                for _ in range(len(x.shape) - 2):
                    token_positions = token_positions.unsqueeze(0)
            
            # Apply RoPE to queries and keys (but not values)
            # Current q, k shape: ... num_heads seq_len d_k
            # RoPE expects shape: ... seq_len d_k and token_positions: ... seq_len
            # We apply RoPE independently to each head, but all heads share the same positions
            
            # Get original shape for later reconstruction
            original_shape = q.shape  # ... num_heads seq_len d_k
            num_heads = original_shape[-3]
            seq_len_dim = original_shape[-2]
            d_k = original_shape[-1]
            
            # Flatten all dimensions before num_heads into a single batch dimension
            # q shape: ... num_heads seq_len d_k -> batch num_heads seq_len d_k
            # where batch = product of all dimensions before num_heads
            batch_size = 1
            for dim_size in original_shape[:-3]:
                batch_size *= dim_size
            
            # Reshape: ... h s d -> (batch, h, s, d)
            q_reshaped = q.reshape(batch_size, num_heads, seq_len_dim, d_k)
            k_reshaped = k.reshape(batch_size, num_heads, seq_len_dim, d_k)
            
            # Flatten batch and heads: (batch, h, s, d) -> (batch*h, s, d)
            q_flat = q_reshaped.reshape(batch_size * num_heads, seq_len_dim, d_k)
            k_flat = k_reshaped.reshape(batch_size * num_heads, seq_len_dim, d_k)
            
            # Expand token_positions to match
            # token_positions can have shape: seq_len or batch seq_len or 1 seq_len
            # Need: (batch * num_heads) seq_len
            # Each head in the same batch uses the same token positions
            if token_positions.dim() == 1:
                # Shape: seq_len -> 1 seq_len
                token_positions = token_positions.unsqueeze(0)
            
            # Now token_positions shape is: batch_in seq_len
            # Need to expand to batch_size if batch_in < batch_size
            if token_positions.shape[0] < batch_size:
                token_positions = token_positions.expand(batch_size, -1)
            
            # Now expand for all heads: batch seq_len -> (batch * num_heads) seq_len
            token_positions_flat = token_positions.unsqueeze(1).expand(batch_size, num_heads, seq_len_dim).reshape(batch_size * num_heads, seq_len_dim)
            
            # Apply RoPE
            q_flat = self.rope(q_flat, token_positions_flat)
            k_flat = self.rope(k_flat, token_positions_flat)
            
            # Reshape back to original shape
            q = q_flat.reshape(original_shape)
            k = k_flat.reshape(original_shape)
        
        # get the causal mask 
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        
        # apply attention 
        out = self.attention(q,k,v,causal_mask)  # shape: ... num_haed seq_len dimension
        
        # rearrange to cancel head, result shape: ... seq_len d_model
        out = rearrange(out, "... h s d -> ... s (h d)", h=self.num_heads) 
        
        # linear proj
        # return torch.einsum("...s d, d l->...s l",out,self.o_proj)
        return torch.einsum("... s d, v d -> ... s v", out, self.o_proj)
        
        
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head self-attention with RoPE
        self.attention_mh = MultiheadAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        
        # Feed-forward network using SwiGLU
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        
        # Layer normalization (pre-norm style)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)  # Before attention
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)  # Before FFN
    

    def forward(self, x, token_positions=None):
        """
        Pre-norm Transformer block with residual connections.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional token positions for RoPE
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        # x + attention(norm(x))
        attn_out = self.attention_mh(self.ln1(x), token_positions=token_positions)
        x = x + attn_out
        
        # Feed-forward with residual connection
        # x + ffn(norm(x))
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x
    
    
class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        self.softmax = Softmax()
    
    
    def forward(self, x): 
        x = self.embedding(x)
        for layer in self.transformer_blocks:
            x = layer(x)
        x = self.norm(x)
        x = self.linear(x)
        return self.softmax(x, dim=-1)