# 学习笔记：PyO3 与 cppyy (Python 扩展工具)

我正在学习如何通过 Rust 和 C++ 为 Python 编写高性能扩展模块。

## 1. cppyy (C++ 动态绑定)

**核心特点**：
- **JIT (即时编译)**：无需手动预编译，在 Python 运行时通过 Cling 解释器实时处理 C++ 代码。
- **无缝集成**：能直接在 Python 中使用 C++ 标准库（如 `std::vector`）和自定义类。
- **快速原型**：适合需要快速测试 C++ 逻辑的场景。

**代码思路演示**：
```python
import cppyy

# 直接定义 C++ 代码
cppyy.cppdef("""
float fast_add(float a, float b) {
    return a + b;
}
""")

# 像调用原生 Python 函数一样使用
from cppyy.gbl import fast_add
print(fast_add(1.2, 3.4))
```

## 2. PyO3 (Rust 扩展模块)

**核心特点**：
- **内存安全**：继承了 Rust 的所有权系统，能有效避免 C++ 中常见的内存溢出或段错误。
- **工业级性能**：经过预编译，运行效率极高，且与 Python 交互的开销极低。
- **构建流**：通常配合 `maturin` 或 `setuptools-rust` 使用。

**开发流程思路**：
1. **Rust 端**：使用 `#[pyfunction]` 和 `#[pymodule]` 宏定义导出接口。
2. **构建端**：运行 `maturin develop` 将 Rust 代码编译并安装到当前的 Python 环境。
3. **Python 端**：直接使用 `import` 导入模块。

## 3. 总结与对比

| 特性 | cppyy | PyO3 |
| :--- | :--- | :--- |
| **底层语言** | C++ | Rust |
| **编译时机** | 运行时 (JIT) | 预编译 (AOT) |
| **易用性** | 极高（无需独立构建步骤） | 中（需配套 Rust 构建工具链） |
| **安全性** | 依赖开发者保证 | 语言层面提供内存安全保证 |

**应用场景**：在深度学习框架（如 PyTorch）、高性能物理引擎或大规模数据处理任务中，这类工具被广泛用于编写计算密集型的底层逻辑。
