import torch
from torch import nn

class SimpleResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x)) + x


class ResNetWithScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.list = nn.ModuleList([SimpleResBlock() for _ in range(3)])
        self.scale = nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, x):
        for layer in self.list:
            x = layer(x)
        return x * self.scale
    
if __name__ == '__main__':
    model = ResNetWithScale()
    optimizer = torch.optim.SGD([model.scale], lr=0.01)
    
    print(f"所有的模型参数数量: {len(list(model.parameters()))}")
    print(f"更新前 scale: {model.scale.item():.4f}")
    
    input = torch.randn(5, 10)
    target = torch.randn(5, 10)
    
    optimizer.zero_grad()
    output = model(input)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"更新后 scale: {model.scale.item():.4f}")
    print("任务检测完成。")
    
