import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def _build_param_keys(n_layer: int = 6) -> list[str]:
    keys = [
        "token_embedding_table.weight",
        "position_embedding_table.weight",
    ]
    for l in range(n_layer):
        keys.extend(
            [
                f"blocks.{l}.ln1.weight",
                f"blocks.{l}.ln1.bias",
                f"blocks.{l}.attn.c_attn.weight",
                f"blocks.{l}.attn.c_proj.weight",
                f"blocks.{l}.attn.c_proj.bias",
                f"blocks.{l}.ln2.weight",
                f"blocks.{l}.ln2.bias",
                f"blocks.{l}.mlp.c_fc.weight",
                f"blocks.{l}.mlp.c_fc.bias",
                f"blocks.{l}.mlp.c_proj.weight",
                f"blocks.{l}.mlp.c_proj.bias",
            ]
        )
    keys.extend(
        [
            "ln_f.weight",
            "ln_f.bias",
            "lm_head.weight",
            "lm_head.bias",
        ]
    )
    return keys


PARAM_KEYS: list[str] = _build_param_keys(6)


class ReferenceCausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class ReferenceMLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x)
        x = self.c_proj(x)
        return x


class ReferenceBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = ReferenceCausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = ReferenceMLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ReferenceGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 65,
        block_size: int = 256,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [ReferenceBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


def _sample_batches(
    data: torch.Tensor, num_batches: int, batch_size: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty((num_batches, batch_size, block_size), dtype=np.int32)
    ys = np.empty((num_batches, batch_size, block_size), dtype=np.int32)
    data_np = data.numpy().astype(np.int32)
    max_start = len(data_np) - block_size
    for b in range(num_batches):
        ix = torch.randint(max_start, (batch_size,)).numpy()
        for i, start in enumerate(ix):
            xs[b, i] = data_np[start : start + block_size]
            ys[b, i] = data_np[start + 1 : start + block_size + 1]
    return xs, ys


def generate_golden_artifacts(
    output_dir: str,
    num_steps: int = 5000,
    eval_iters: int = 50,
    seed: int = 998244353,
    data_path: str = "data/input.txt",
) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    resolved_data = Path(data_path)
    if not resolved_data.exists():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / data_path
        if candidate.exists():
            resolved_data = candidate
        else:
            fallback = repo_root / "ng-video-lecture" / "input.txt"
            if fallback.exists():
                resolved_data = fallback

    text = resolved_data.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    assert vocab_size == 65, f"Expected vocab_size=65, got {vocab_size}"

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encoded = [stoi[c] for c in text]
    data = torch.tensor(encoded, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 64
    block_size = 256
    n_embd = 384
    n_head = 6
    n_layer = 6

    model = ReferenceGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    )
    num_parameters = sum(p.numel() for p in model.parameters())
    assert (
        num_parameters == 10788929
    ), f"Expected 10788929 parameters, got {num_parameters}"

    # Export initial weights theta_0
    state_dict = model.state_dict()
    weights_dict: dict[str, np.ndarray] = {}
    flat_weights: list[np.ndarray] = []
    for k in PARAM_KEYS:
        arr = state_dict[k].detach().cpu().numpy().astype(np.float32)
        weights_dict[k] = arr
        flat_weights.append(arr.reshape(-1))

    np.savez(out_path / "golden_weights.npz", **weights_dict)
    np.concatenate(flat_weights).astype(np.float32).tofile(
        out_path / "golden_weights.bin"
    )

    # Pre-sample deterministic batches
    train_x, train_y = _sample_batches(train_data, num_steps, batch_size, block_size)
    eval_train_x, eval_train_y = _sample_batches(
        train_data, eval_iters, batch_size, block_size
    )
    val_x, val_y = _sample_batches(val_data, eval_iters, batch_size, block_size)
    infer_prompt_b1 = np.zeros((1, 1), dtype=np.int32)
    infer_prompt_b64 = np.zeros((batch_size, 1), dtype=np.int32)

    np.savez(
        out_path / "golden_batches.npz",
        train_x=train_x,
        train_y=train_y,
        eval_train_x=eval_train_x,
        eval_train_y=eval_train_y,
        val_x=val_x,
        val_y=val_y,
        infer_prompt_b1=infer_prompt_b1,
        infer_prompt_b64=infer_prompt_b64,
    )

    concat_batches = np.concatenate(
        [
            train_x.reshape(-1),
            train_y.reshape(-1),
            eval_train_x.reshape(-1),
            eval_train_y.reshape(-1),
            val_x.reshape(-1),
            val_y.reshape(-1),
        ]
    ).astype(np.int32)
    concat_batches.tofile(out_path / "golden_batches.bin")

    # Compute reference step 0 and step 1 losses with AdamW
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    x0 = torch.from_numpy(train_x[0]).long()
    y0 = torch.from_numpy(train_y[0]).long()
    _, loss0 = model(x0, y0)
    loss_step_0 = float(loss0.item())

    optimizer.zero_grad(set_to_none=True)
    loss0.backward()
    optimizer.step()

    idx_next = 1 if num_steps > 1 else 0
    x1 = torch.from_numpy(train_x[idx_next]).long()
    y1 = torch.from_numpy(train_y[idx_next]).long()
    with torch.no_grad():
        _, loss1 = model(x1, y1)
    loss_step_1 = float(loss1.item())

    meta = {
        "seed": seed,
        "num_parameters": num_parameters,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "num_steps": num_steps,
        "eval_iters": eval_iters,
        "loss_step_0": loss_step_0,
        "loss_step_1": loss_step_1,
        "itos": {str(k): v for k, v in itos.items()},
        "stoi": stoi,
    }

    with open(out_path / "golden_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deterministic golden reference weights and batches."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/golden",
        help="Directory to write golden artifacts.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5000,
        help="Number of training steps to pre-sample.",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=50,
        help="Number of evaluation batches per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=998244353,
        help="Global random seed.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/input.txt",
        help="Path to input.txt dataset.",
    )
    args = parser.parse_args()
    meta = generate_golden_artifacts(
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        eval_iters=args.eval_iters,
        seed=args.seed,
        data_path=args.data_path,
    )
    print(
        f"Golden artifacts generated in {args.output_dir}: "
        f"params={meta['num_parameters']}, "
        f"loss_step_0={meta['loss_step_0']:.6f}, "
        f"loss_step_1={meta['loss_step_1']:.6f}"
    )
