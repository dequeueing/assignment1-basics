我需要了解torch的这些抽象：

• torch.nn.Parameter
• Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)
• The torch.optim.Optimizer base class

### 关于 super().__init__()
为了激活 PyTorch 对模型状态的管理机制。基类的 `__init__` 方法会创建几个内部的“账本”（即私有字典）：
*   `_parameters`: 存放所有的 `nn.Parameter`。
*   `_modules`: 存放子模块（如 `nn.Linear`, `nn.Sequential`）。
*   `_buffers`: 存放不需要梯度的张量（如 BatchNorm 的均值）。

如果没有调用 `super().__init__()`，当你尝试分配 `nn.Parameter` 给 `self` 时，PyTorch 会因为找不到这些内部字典而报错。

### 关于 forward 方法
*   `__init__` 负责初始化参数和子模块（“定义零件”）。
*   `forward` 负责定义数据流（“组装流水线”）。
*   当你执行 `output = model(input)` 时，PyTorch 内部会自动调用 `forward` 函数。如果不定义，将无法进行计算。

### 关于 torch.optim.Optimizer
优化器负责根据梯度更新模型的 `Parameter`。

**核心工作流程：**
1.  `optimizer.zero_grad()`: 清空旧的梯度。这是必要的，因为 PyTorch 默认会累加梯度。
2.  `loss.backward()`: 计算损失函数相对于模型参数的梯度。
3.  `optimizer.step()`: 根据优化算法（如 SGD, AdamW）更新参数。

**什么是 param_groups？**
`optimizer.param_groups` 是一个包含字典的列表。每个字典代表一组参数及其相关的超参数（如 `lr`, `weight_decay`）。这允许我们为模型不同的部分设置不同的学习率（即“差异化学习率”）。

### nn.Sequential vs nn.ModuleList
*   **nn.Sequential**：
    *   **优点**：代码简洁，内置了 `forward` 逻辑，数据自动流经各层。
    *   **缺点**：不灵活。只能实现简单的线性顺序结构，难以实现残差连接（Skip Connection）或分支结构。
*   **nn.ModuleList**：
    *   **优点**：非常灵活。仅用于“注册”层，让你能在 `forward` 中通过循环或其他逻辑自定义数据流。
    *   **与 python list 的区别**：普通的 `list` 无法让 PyTorch 追踪到内部层的参数，而 `ModuleList` 可以。当你调用 `model.to(device)` 时，`ModuleList` 里的层也会跟着移动。
    *   **应用场景示例：Transformer 的 Block**
        在 Transformer 或残差网络（ResNet）中，每一层的输出通常要和输入相加（残差连接）：
        $$x = layer(x) + x$$
        这种结构在 `nn.Sequential` 中很难直接实现，但在 `nn.ModuleList` 的循环中可以轻松完成。


### 常见坑点：类 vs 实例 (Class vs Instance)
在使用层（如同 `nn.ReLU`, `nn.Linear`）时，务必在 `__init__` 中进行实例化（即加括号 `()`）：
*   `self.relu = nn.ReLU`：错误！这只是把类本身赋值给了变量。
*   `self.relu = nn.ReLU()`：正确！这创建了一个可以被调用的对象。
如果不加括号，在 `forward` 中调用它会返回一个模块对象而非计算结果，导致无法与 Tensor 相加。

### 总结：PyTorch 是如何管理参数的？
*   **层级结构**：PyTorch 会递归地搜索所有通过 `self.xxx = ...` 赋值的属性。
*   **注册机制**：如果属性是 `nn.Parameter` 或 `nn.Module`（包含 `Sequential` 和 `ModuleList`），它就会被登记在册。
*   **递归搜索**：`model.parameters()` 会找出主模型、子模块、以及子模块的子模块中所有的参数。

Parameter: 自动注册的 Tensor，优化器会更新它。
ModuleList vs list: 前者能让 PyTorch 追踪到内部的参数。
