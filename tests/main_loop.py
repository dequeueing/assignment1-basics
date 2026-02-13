from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class TrainConfig:
    max_iters: int = 5000      # 总迭代步数
    eval_interval: int = 500   # 每隔多少步评估一次
    save_interval: int = 1000  # 每隔多少步保存一次
    learning_rate: float = 6e-4
    batch_size: int = 32
    device: str = "cuda"
    context_length: int = 32
    # TODO: more hyperparameters to come
    
config = TrainConfig()

from .training import DataLoader
dataloader = DataLoader(config.batch_size,context_length=config.context_length,device=config.device)
# dataloader.process_data()


def train():
    # Init model and optimizer
    from .modules import TransformerLM
    from .training import AdamWGD
    from .training import save_checkpoint
    lm = TransformerLM().to(config.device)
    optimizer = AdamWGD(lm)
    
    
    for iter in range(config.max_iters):
        xb, yb = dataloader.process_data()(train_data, ...)
        
        logits, loss = lm(xb, targets=yb)
        
        # backward and step
        optimizer.zero_grad(set_to_none=True) # set_to_none=True 通常更快
        loss.backward()
        # TODO: clip gradient 
        optimizer.step()
        # TODO: update scheduler 
        
        if iter % config.eval_interval == 0:
            pass
            # 这里的 estimate_loss 是一个辅助函数，
            # 专门在 val_data 上跑几百个 batch 算平均 loss，不进行反向传播
            # losses = estimate_loss(model, train_data, val_data)
            # print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        if iter % config.save_interval == 0:
            save_checkpoint(lm, optimizer, iter, f"ckpt_{iter}.pt")
        
    
if __name__ == '__main__':
    # TODO: tokenize user texts and save them

    # 假设我们要加载 train.bin
    # mode='r' 表示只读，避免误修改数据
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    
    
    train()
    
    