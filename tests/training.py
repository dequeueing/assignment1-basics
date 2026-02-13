import torch
from torch import Tensor
from einops import rearrange
from collections.abc import Callable, Iterable
from jaxtyping import Float, Int
from typing import Optional
import math



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
    
    
class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params,defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']  # get the learning rate
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data 
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t+1
                
        return loss
                
if __name__ == "__main__":
    for lr in [1, 10, 100]:
        print(f"\n--- Testing lr={lr} ---")
        weight = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weight], lr=lr)
        print(f"\nThe current lr: {lr}")
        
        for t in range(10):
            opt.zero_grad()
            loss = (weight**2).mean()
            print(f"Iteration {t}, Loss: {loss.item():.6f}")
            loss.backward()
            opt.step()
    