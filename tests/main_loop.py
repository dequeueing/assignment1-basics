"""
Training loop for a small Transformer language model.
Uses byte-level tokenization (vocab_size=256) on TinyStories data for quick testing.

Usage:
    cd /home/exouser/assignment1-basics
    uv run python -m tests.main_loop
"""

import os
import math
import time
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ---- Model ----
    vocab_size: int = 256          # byte-level tokenization: 每个 byte 就是一个 token
    context_length: int = 128
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    rope_theta: float = 10000.0

    # ---- Training ----
    max_iters: int = 500
    eval_interval: int = 50        # 每隔多少步评估
    eval_iters: int = 20           # 评估时跑多少个 batch 取平均
    save_interval: int = 200       # 每隔多少步保存 checkpoint
    log_interval: int = 10         # 每隔多少步打印 loss
    batch_size: int = 32
    device: str = "cuda"

    # ---- Optimizer / Schedule ----
    max_learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_iters: int = 50
    cosine_cycle_iters: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ---- Data ----
    data_dir: str = "/home/exouser/assignment1-basics/data"
    train_file: str = "TinyStoriesV2-GPT4-train.txt"
    val_file: str = "TinyStoriesV2-GPT4-valid.txt"
    max_train_bytes: int = 1_000_000   # 1 MB 用于快速测试
    max_val_bytes: int = 100_000       # 100 KB 用于验证

    # ---- Output ----
    out_dir: str = "out"


# ---------------------------------------------------------------------------
# 数据准备: byte-level tokenization (每个 byte 就是一个 token, vocab_size=256)
# ---------------------------------------------------------------------------

def prepare_data(config: TrainConfig):
    """读取原始文本，按 byte 转为 uint16 numpy 数组保存到 .bin 文件。"""
    os.makedirs(config.out_dir, exist_ok=True)

    train_bin = os.path.join(config.out_dir, "train.bin")
    val_bin = os.path.join(config.out_dir, "val.bin")

    if not os.path.exists(train_bin):
        train_path = os.path.join(config.data_dir, config.train_file)
        print(f"准备训练数据: {train_path} (前 {config.max_train_bytes / 1e6:.1f} MB) ...")
        with open(train_path, "rb") as f:
            raw = f.read(config.max_train_bytes)
        tokens = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        tokens.tofile(train_bin)
        print(f"  保存 {len(tokens):,} tokens -> {train_bin}")

    if not os.path.exists(val_bin):
        val_path = os.path.join(config.data_dir, config.val_file)
        print(f"准备验证数据: {val_path} (前 {config.max_val_bytes / 1e6:.1f} MB) ...")
        with open(val_path, "rb") as f:
            raw = f.read(config.max_val_bytes)
        tokens = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        tokens.tofile(val_bin)
        print(f"  保存 {len(tokens):,} tokens -> {val_bin}")

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def forward_logits(model, x):
    """跳过 TransformerLM 末尾的 softmax，直接返回 logits — 数值更稳定。"""
    x = model.embedding(x)
    for layer in model.transformer_blocks:
        x = layer(x)
    x = model.norm(x)
    logits = model.linear(x)
    return logits


def cross_entropy_loss(logits, targets):
    """从 logits 计算 cross-entropy loss (用 log-sum-exp 技巧保证数值稳定)。

    logits:  (B, T, V)
    targets: (B, T)
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T).long()

    max_val = logits_flat.max(dim=-1, keepdim=True).values
    shifted = logits_flat - max_val
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted[torch.arange(B * T, device=logits.device), targets_flat]
    loss = (log_sum_exp - target_logits).mean()
    return loss


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, dataloader, config):
    """在 train / val 上各跑 eval_iters 个 batch，取平均 loss。"""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(config.eval_iters):
            xb, yb = dataloader.process_data(data)
            logits = forward_logits(model, xb)
            total += cross_entropy_loss(logits, yb).item()
        losses[name] = total / config.eval_iters
    model.train()
    return losses


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(config.out_dir, exist_ok=True)

    # ---- 数据 ----
    train_data, val_data = prepare_data(config)
    print(f"训练 tokens: {len(train_data):,}  |  验证 tokens: {len(val_data):,}")

    # ---- 模型 ----
    from tests.modules import TransformerLM
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # ---- 优化器 & 工具 ----
    from tests.training import (
        AdamWGD,
        DataLoader,
        cosine_annealing_schedule,
        gradient_clipping,
        save_checkpoint,
    )

    optimizer = AdamWGD(
        model.parameters(),
        lr=config.max_learning_rate,
        weight_decay=config.weight_decay,
    )
    dataloader = DataLoader(
        batch_size=config.batch_size,
        context_length=config.context_length,
        device=config.device,
    )

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"开始训练  iters={config.max_iters}  batch={config.batch_size}  ctx={config.context_length}")
    print(f"模型  d={config.d_model}  layers={config.num_layers}  heads={config.num_heads}  d_ff={config.d_ff}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(config.max_iters):
        # 1) 学习率调度: cosine annealing with warmup
        lr = cosine_annealing_schedule(
            it,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 2) 定期评估
        if it % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, dataloader, config)
            elapsed = time.time() - t0
            print(
                f"[iter {it:4d} | {elapsed:6.1f}s]  "
                f"train_loss={losses['train']:.4f}  val_loss={losses['val']:.4f}  lr={lr:.2e}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, optimizer, it, os.path.join(config.out_dir, "best.pt"))

        # 3) 定期保存 checkpoint
        if it > 0 and it % config.save_interval == 0:
            save_checkpoint(model, optimizer, it, os.path.join(config.out_dir, f"ckpt_{it}.pt"))

        # 4) 取一个训练 batch
        xb, yb = dataloader.process_data(train_data)

        # 5) 前向传播 (直接拿 logits，不过 softmax)
        logits = forward_logits(model, xb)
        loss = cross_entropy_loss(logits, yb)

        # 6) 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 7) 梯度裁剪
        gradient_clipping(model.parameters(), config.max_grad_norm)

        # 8) 参数更新
        optimizer.step()

        # 9) 定期打印训练 loss
        if it > 0 and it % config.log_interval == 0 and it % config.eval_interval != 0:
            elapsed = time.time() - t0
            tps = (it * config.batch_size * config.context_length) / elapsed
            print(f"  [iter {it:4d}]  loss={loss.item():.4f}  lr={lr:.2e}  tok/s={tps:,.0f}")

    # ---- 训练结束 ----
    losses = estimate_loss(model, train_data, val_data, dataloader, config)
    elapsed = time.time() - t0
    print(f"\n[完成 | {elapsed:.1f}s]  train_loss={losses['train']:.4f}  val_loss={losses['val']:.4f}")
    save_checkpoint(model, optimizer, config.max_iters, os.path.join(config.out_dir, "final.pt"))
    print(f"Checkpoint 已保存到 {config.out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = TrainConfig()

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        config.device = "cpu"

    train(config)
    
    