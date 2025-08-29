from __future__ import annotations

from collections import defaultdict
import os
from typing import IO, Any, BinaryIO, Dict, List, Tuple
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab: Dict[int, bytes] = {}
    merges: List[Tuple[bytes, bytes]] = []
    
    # Init vocabulary
    for i in range(256):
        i_bytes = bytes([i])
        vocab[i] = i_bytes
        
    # Add special tokens 
    vocab_len = len(vocab)
    for sp in special_tokens:
        sp_bytes = sp.encode('utf-8')
        vocab[vocab_len] = sp_bytes
        vocab_len += 1
        
    # Debug print the vocab
    # for key in vocab.keys():
    #     print(f"token id: {key}, token: {vocab[key]}")
        
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        
        # data structure for mering 
        # 1. byte_pair_cnt: {(b'T', b'h'): 2, ...}. Find the byte pair with highest count
        # 2. pretoken_cnt: {'The': 2, 'That': 3}
        # 3. pretoken_bytes: {'The': [b'T', b'h', b'e']}
        # 4. bytes_pair_pretoken: {(b'T', b'h'): ['The', 'That']}
        byte_pair_cnt: Dict[tuple[bytes, bytes], int] = defaultdict(int)
        pretoken_cnt: Dict[str, int] = defaultdict(int)
        pretoken_bytes: Dict[str, list[bytes]] = defaultdict(list)
        bytes_pair_pretoken: Dict[tuple[bytes, bytes], list[str]] = defaultdict(list)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            
            # print(type(chunk)) # str
            # print(chunk[:50])
            
            # Remove special token
            import re
            escaped_tokens = [re.escape(token) for token in special_tokens]
            special_tokens_pattern = '|'.join(escaped_tokens)
            documents = re.split(special_tokens_pattern, chunk)
            # print(f"len first few result after removing: {len(result)}")  # each item in result is a document
            # print(f"type few result after removing: {type(result[0])}")
            # print(f"type few result after removing: {result[0]}")
            
            
            # Pre-token each ducoment, init data sutrctures for merging
            import regex as re
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for document in documents:
                # pretokenized_doc = re.findall(PAT, document)
                # print(f"pre-token list (first 10): {pretokenized_doc[:10]}")
                # for pretoken in pretokenized_doc:
                #     # TODO: each pretoken to each byte-pair in it
                #     pass
                pretokens = re.finditer(PAT, document)
                for match in pretokens:
                    pretoken = match.group()
                    pretoken_cnt[pretoken] += 1  # count pretoken
                    byte_seq = pretoken.encode('utf-8')
                    if pretoken_cnt[pretoken] == 1:  # first time seeing this pretoken
                        pretoken_bytes[pretoken] = []
            
                    for i in range(len(byte_seq) - 1):
                        first = bytes([byte_seq[i]])
                        second = bytes([byte_seq[i+1]])
                        byte_pair = (first, second)
                        byte_pair_cnt[byte_pair] += 1
                        if not bytes_pair_pretoken[byte_pair]:
                            bytes_pair_pretoken[byte_pair] = []
                        
                        if pretoken_cnt[pretoken] == 1: # first time seeing this pretoken
                            bytes_pair_pretoken[byte_pair].append(pretoken)
                            pretoken_bytes[pretoken].append(bytes([byte_seq[i]]))
                            if i == len(byte_seq) - 2:
                                pretoken_bytes[pretoken].append(bytes([byte_seq[i+1]]))
                    # print(f"pretoken: {pretoken}, byteseq: {byte_seq}, type of seq: {type(byte_seq)}")
            
            print(f"byte_pair_cnt: {byte_pair_cnt}")
            print(f"pretoken_cnt: {pretoken_cnt}")
            print(f"pretoken_bytes: {pretoken_bytes}")
            print(f"bytes_pair_pretoken: {bytes_pair_pretoken}")
            
            # save pretoken cnt locally
            import json
            file_path = 'pretoken_cnt.txt'
            with open(file_path, "w") as json_file:
                json.dump(pretoken_cnt, json_file, indent=4)

            
        # Test: merge for once
        # one itertaion
        # pretokens: ['the' 'that' 'these']
        # init:
        # pretoken_cnt: {'the': 2, 'that': 1, 'these':1}
        # pretoken_bytes: {'the': [b't', b'h', b'e'], 'that': [b't', b'h', b'a', b't'}
        # byte_pair_cnt: {(b't', b'h'): 2, (b'h', b'e'): 1, ...}
        # byte_pair_pretoken: {(b't', b'h'): ['the', 'that', 'these'], ...}
        
        # Find the most frequent byte pair: (b't', b'h')
        # pretokens related to this byte pair:  ['the', 'that', 'these']
        # update the bytes sequence of these tokens: 'the': [b'th', b'e']
        # The byte pair that will be affected: (b'h', b'e') -> (b'th', b'e'), update the cnt of (b'h', b'e')
        while(len(vocab) < vocab_size):
        
            highest_cnt = max(byte_pair_cnt.values())
            byte_pairs_with_highest_cnt = [key for key in byte_pair_cnt.keys() if byte_pair_cnt[key] == highest_cnt]
            bytes_pair_to_merge = max(byte_pairs_with_highest_cnt)
            merged_bytes = bytes_pair_to_merge[0] + bytes_pair_to_merge[1]
            # print(f"bytes_pair_to_merge: {bytes_pair_to_merge}")
            
            # debug: compare [(b" t", b"he"), (b"r", b"e")]
            p1 = (b"r", b"e")
            p2 = (b" t", b"he")
            if bytes_pair_to_merge == p1:
                print(f'{p1} cnt = {byte_pair_cnt[p1]}')
                print(f'{p2} cnt = {byte_pair_cnt[p2]}')
            
            # prepare new byte_pretoken mapping
            bytes_pair_pretoken[merged_bytes] = []
            
            # Remove byte pair
            # print(f"bytes_pair_to_merge: {bytes_pair_to_merge}")
            byte_pair_cnt.pop(bytes_pair_to_merge)
            
            # update each token
            pretokens_to_update = bytes_pair_pretoken[bytes_pair_to_merge]
            # print(f"pretokens_to_update: {pretokens_to_update}")
            for pretoken in pretokens_to_update:
                sub_sequence = [bytes_pair_to_merge[0], bytes_pair_to_merge[1]]
                    
                bytes_sequence = pretoken_bytes[pretoken]
                # print(f"bytes_sequence for {pretoken}: {bytes_sequence}")
                # print(f"sub_sequence looking for: {sub_sequence}")
                # print(f"sub_sequence to find: {sub_sequence}")
                
                # find the sub-sequence in the bytes_sequence
                index_found = None
                for i in range(len(bytes_sequence) - 1):
                    seq1 = bytes_sequence[i:i+2]
                    seq2 = sub_sequence
                    # print(f"\nseq1: {seq1}")
                    # print(f"seq2: {seq2}\n")
                    if bytes_sequence[i:i+2] == sub_sequence:
                        index_found = i
                        break  # Found it, so we can stop
                if index_found is None:
                    print(f"!bytes_sequence for {pretoken}: {bytes_sequence}")
                    print(f"!sub_sequence looking for: {sub_sequence}")
                    raise Exception('subsequence not found in the complete bytes sequence')
                
                # Extract the four bytes to update 
                starting_index = max(index_found -1, 0)
                ending_index = index_found + 3
                bytes_to_update = bytes_sequence[starting_index :ending_index]
                
                # expected: a new list of bytes
                new_byte_pairs = []
                new_byte_sequence = []
                bytes_pair_pretoken[bytes_pair_to_merge].remove(pretoken) # remove byte_pretoken mapping
                if len(bytes_to_update) == 4:
                    pair1 = (bytes_to_update[0], bytes_to_update[1])
                    pair2 = (bytes_to_update[2], bytes_to_update[3])
                    # print(f"pair1: {pair1}, pair2: {pair2}")
                    before = byte_pair_cnt[pair1]
                    byte_pair_cnt[pair1] -= pretoken_cnt[pretoken]
                    byte_pair_cnt[pair2] -= pretoken_cnt[pretoken]
                    after = byte_pair_cnt[pair1]
                    
                    # debug print 
                    # print(f"updating pretoken: {pretoken}, pair: {pair1}")
                    # print(f"before decrementing: {before}, after: {after}")
                    
                    bytes_pair_pretoken[pair1].remove(pretoken)
                    bytes_pair_pretoken[pair2].remove(pretoken)
                    
                    new_byte_pair1 = (bytes_to_update[0], merged_bytes)
                    new_byte_pair2 = (merged_bytes, bytes_to_update[-1])
                    new_byte_pairs.append(new_byte_pair1)
                    new_byte_pairs.append(new_byte_pair2)
                    
                    new_byte_sequence.append(bytes_to_update[0])
                    new_byte_sequence.append(merged_bytes)
                    new_byte_sequence.append(bytes_to_update[-1])
                elif len(bytes_to_update) == 3:
                    pair1 = (bytes_to_update[0], bytes_to_update[1])
                    pair2 = (bytes_to_update[1], bytes_to_update[2])
                    assert(pair1 == bytes_pair_to_merge or pair2 == bytes_pair_to_merge)
                    # print(f"pair1: {pair1}, pair2: {pair2}")
                    if pair1 == bytes_pair_to_merge:
                        byte_pair_cnt[pair2] -= pretoken_cnt[pretoken]
                        if pretoken in bytes_pair_pretoken[pair2]:
                            bytes_pair_pretoken[pair2].remove(pretoken)
                        else:
                            print(f"pair2: {pair2}")
                            print(f"pretoken: {pretoken}")
                            raise Exception('error removing pretoken from mappings')
                        new_byte_pair = (merged_bytes, bytes_to_update[-1])
                        new_byte_sequence.append(merged_bytes)
                        new_byte_sequence.append(bytes_to_update[-1])
                    else:
                        byte_pair_cnt[pair1] -= pretoken_cnt[pretoken]
                        bytes_pair_pretoken[pair1].remove(pretoken)
                        new_byte_pair = (bytes_to_update[0], merged_bytes)
                        new_byte_sequence.append(bytes_to_update[0])
                        new_byte_sequence.append(merged_bytes)
                    new_byte_pairs.append(new_byte_pair)
                else:
                    if len(bytes_to_update) != 2:
                        print(f"!len(bytes_to_update): {len(bytes_to_update)}")
                        print(f"bytes_to_update: {bytes_to_update}")
                        raise Exception('bytes_to_update having trouble!')
                    
                    new_byte_sequence.append(merged_bytes)
                # print(f"new_byte_pairs: {new_byte_pairs}")
                # print(f"new_byte_sequence: {new_byte_sequence}")
                
                # increment new_byte_pairs
                for byte_pair in new_byte_pairs:
                    byte_pair_cnt[byte_pair] += pretoken_cnt[pretoken]
                    bytes_pair_pretoken[byte_pair].append(pretoken)
                    
                
                # replace pretoken's bytes
                start_index = -1
                for i in range(len(bytes_sequence) - len(bytes_to_update) + 1):
                    if bytes_sequence[i : i + len(bytes_to_update)] == bytes_to_update:
                        start_index = i
                        break

                # 2. Use slicing to replace the sub-sequence
                if start_index != -1:
                    end_index = start_index + len(bytes_to_update)
                    bytes_sequence[start_index:end_index] = new_byte_sequence

                # print(f"replaced bytes_sequence: {bytes_sequence}")
                
                
            # Update vocabulary and merges
            vocab[len(vocab)] = merged_bytes
            merges.append(bytes_pair_to_merge)
            # print(f"merges: {merges}")
            # print(f"updated vocab:\n{vocab}")
            # print(f"bytes_pair_to_merge: {bytes_pair_to_merge}")

    return vocab, merges
            
            
            
            
if __name__ == '__main__':
    input_path = "/home/exouser/cs336/assignment1-basics/tests/adatper_main.txt"
    vocab, merges  = run_train_bpe(
        input_path=input_path,
        vocab_size=260,
        special_tokens=["<|endoftext|>"],
    )
    print('==================================')
    print(vocab)
    print(merges)
