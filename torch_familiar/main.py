import torch

def einsum_examples():
    # 1. 矩阵乘法 (Matrix Multiplication)
    # A: (2, 3), B: (3, 4) -> C: (2, 4)
    # 规则：j 是收缩维度（在输入中重复出现，在输出中消失）
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = torch.einsum('ij,jk->ik', A, B)
    print(f"Matrix Mult shape: {C.shape}")

    # 2. 批量矩阵乘法 (Batch Matrix Multiplication)
    # b 为 batch 维度，i, j, k 同上
    A_batch = torch.randn(10, 2, 3)
    B_batch = torch.randn(10, 3, 4)
    C_batch = torch.einsum('bij,bjk->bik', A_batch, B_batch)
    print(f"Batch Matrix Mult shape: {C_batch.shape}")

    # 3. 计算 Trace (对角线元素之和)
    # 规则：重复的字母 i 在输出中消失，表示对该维度求和
    square_mat = torch.randn(5, 5)
    trace = torch.einsum('ii->', square_mat)
    print(f"Trace: {trace.item():.4f}, torch.trace: {torch.trace(square_mat).item():.4f}")

    # 4. 转置 (Transpose)
    # 规则：通过改变字母顺序重新排列维度
    transposed = torch.einsum('ij->ji', A)
    print(f"Original shape: {A.shape}, Transposed shape: {transposed.shape}")

    # 5. 向量外积 (Outer Product)
    v1 = torch.randn(3)
    v2 = torch.randn(4)
    outer = torch.einsum('i,j->ij', v1, v2)
    print(f"Outer product shape: {outer.shape}")

    # 6. 使用省略号 (...) 处理多余维度
    # 假设我们只想对最后两个维度做转置，不管前面有多少个 batch 维度
    high_dim = torch.randn(2, 5, 10, 3)
    # ... 代表 (2, 5)，我们只交换最后两个 (10, 3 -> 3, 10)
    swapped = torch.einsum('...ij->...ji', high_dim)
    print(f"High-dim original: {high_dim.shape}, Swapped: {swapped.shape}")

from einops import rearrange


def einops_examples():
    print("\n--- Einops Rearrange Examples ---")
    # 模拟 Transformer 中的多头拆分
    # b: batch, s: seq_len, h: num_heads, d: head_dim
    # 假设输入是 (batch, seq_len, hidden_dim) 其中 hidden_dim = num_heads * head_dim
    batch, seq_len, num_heads, head_dim = 2, 10, 8, 64
    x = torch.randn(batch, seq_len, num_heads * head_dim)
    
    # 1. 拆分维度 (Split)
    # 语义：把最后一个维度拆成 (h d)
    x_heads = rearrange(x, 'b s (h d) -> b s h d', h=num_heads)
    print(f"Split: {x.shape} -> {x_heads.shape}")

    # 2. 转置 (Transpose/Permute)
    # Transformer 中通常需要把 head 维度放到 seq 维度前面
    x_transposed = rearrange(x_heads, 'b s h d -> b h s d')
    print(f"Transpose: {x_heads.shape} -> {x_transposed.shape}")

    # 3. 合并维度 (Flatten)
    # 处理完后，把多头合并回原来的形状
    x_merged = rearrange(x_transposed, 'b h s d -> b s (h d)')
    print(f"Merge: {x_transposed.shape} -> {x_merged.shape}")

    # 4. 图像转 Patch (Vision Transformer 核心)
    # 假设一张 224x224 的 RGB 图像，切成 16x16 的 patch
    img = torch.randn(1, 3, 224, 224)
    # p1, p2 是 patch 的高和宽
    patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
    print(f"Image to Patches: {img.shape} -> {patches.shape}")
    

def transformer():
    # x: (batch=1, seq=128, hidden=512)
    batch=1
    seq_len=128
    hidden=512
    # split into 8 heads
    num_heads = 8
    head_dim = 64
    x = torch.randn(batch, seq_len, hidden)
    
    x = rearrange(x, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=num_heads)
    key = x = torch.randn(batch, num_heads, seq_len, head_dim)
    
    # query and key's dot product 
    result = torch.einsum('b h q d, b h k d -> b h q k', x, key)
    print("ok")
    

if __name__ == "__main__":
    # 为了演示，我们可以把之前的函数也跑一下
    # einsum_examples()
    # einops_examples()
    transformer()
