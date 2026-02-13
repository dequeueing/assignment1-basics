import torch
from torch import Tensor
from einops import rearrange
from jaxtyping import Float, Int



class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]): 
        # Note:
        # pay attention to the dimension, think whether we should keep this dim or not
        # when indexing, still make sure the dim is correct
        # some functions we need to specify the dim to operate along. 
        
        # get the maximum 
        max_val = torch.max(inputs, dim=-1, keepdim=True).values
        
        # minus the maximum
        shifted_inputs = inputs - max_val
        
        # get exp and sum 
        sum_exp = torch.sum(torch.exp(shifted_inputs),dim=-1) 
        log_sum_exp = torch.log(sum_exp)  # shape: (batch_size)
        
        # get target    
        batch_size = inputs.shape[0]
        row_indices = torch.arange(batch_size, device=inputs.device)
        target_logits = shifted_inputs[row_indices, targets]  # shape: (batch_size)
        
        # calculate loss
        loss = log_sum_exp - target_logits
        return torch.mean(loss)