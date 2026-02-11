import torch
import torch.nn as nn

# 1. 练习使用 Parameter
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: 使用 nn.Parameter 定义 weights 和 bias
        # paramter 接受一个 tensor作为模型的参数。
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # 实现线性变换：y = xW^T + b
        # 注意：nn.Linear 内部存储权重的方式通常是 (out_features, in_features)
        return x @ self.weight.t() + self.bias

# 2. 练习使用 Sequential 和 ModuleList
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.module_list = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ])

    def forward(self, x):
        # Sequential 会自动按顺序执行其包含的所有层
        return self.seq(x)

# 3. 观察 Optimizer
if __name__ == "__main__":
    model = MyModel()
    
    # 初始化优化器，传入模型的参数和学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("--- Optimizer param_groups 结构 ---")
    # param_groups 是一个 list，每个元素是一个 dict
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i} keys: {group.keys()}")
        print(f"Group {i} learning rate: {group['lr']}")
        # group['params'] 包含了该组需要优化的所有参数
    
    # 模拟一个训练步骤
    print("\n--- 模拟训练步骤 ---")
    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)
    
    # 1. 清空梯度
    optimizer.zero_grad()
    
    # 2. 前向传播
    output = model(input_data)
    
    # 3. 计算损失
    loss = nn.MSELoss()(output, target)
    print(f"Current Loss: {loss.item()}")
    
    # 4. 反向传播（计算梯度）
    loss.backward()
    
    # 5. 更新参数
    optimizer.step()
    print("参数已更新完成。")
