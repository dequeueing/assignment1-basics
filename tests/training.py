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
    
        
class AdamWGD(torch.optim.Optimizer): 
    def __init__(self,params,lr=1e-3,weight_decay=0.01,betas=(0.9, 0.999),eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr, "weight_decay":weight_decay, "betas":betas, "eps":eps}
        
        super().__init__(params,defaults)
        
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']  # get the learning rate
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            betas = group['betas']  # get the betas
            beta1 = betas[0]
            beta2 = betas[1]
            
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # get gradient 
                grad = p.grad.data 
                
                state = self.state[p]
                t = state.get("t", 1)
                
                # the moment vector estimate 
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.torch.zeros_like(grad))
                
                # update moment vector
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # adjusted learning rate
                lr_adjusted = lr * math.sqrt(1 - math.pow(beta2,t)) / (1 - math.pow(beta1,t))
                
                # update parameters
                p.data -= lr_adjusted * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                
                state['t'] = t+1
                state['m'] = m
                state['v'] = v
                
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
    