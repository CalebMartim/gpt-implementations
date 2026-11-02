import math
import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

    class _TritonShim:
        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def next_power_of_2(n: int) -> int:
            p = 1
            while p < n:
                p <<= 1
            return p

        @staticmethod
        def cdiv(a: int, b: int) -> int:
            return (a + b - 1) // b

    class _TlShim:
        constexpr = int

    triton = _TritonShim()
    tl = _TlShim()


@triton.jit
def _layernorm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Rstd_ptr,
    stride_x,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xbar = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)

    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = xbar * rstd
    y = x_hat * w + b

    y_ptrs = Y_ptr + row * stride_x + cols
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _layernorm_bwd_dx_kernel(
    DX_ptr,
    DY_ptr,
    DW_partial_ptr,
    DB_partial_ptr,
    X_ptr,
    W_ptr,
    Mean_ptr,
    Rstd_ptr,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    dy = tl.load(DY_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = tl.where(mask, w * dy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    tl.store(DX_ptr + row * stride_x + cols, dx, mask=mask)
    tl.store(DW_partial_ptr + row * stride_x + cols, dy * xhat, mask=mask)
    tl.store(DB_partial_ptr + row * stride_x + cols, dy, mask=mask)


@triton.jit
def _layernorm_bwd_dwdb_kernel(
    DW_partial_ptr,
    DB_partial_ptr,
    DW_ptr,
    DB_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    acc_dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_N], dtype=tl.float32)

    for start_m in range(0, M, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & mask_n[None, :]
        offs = rows[:, None] * N + cols[None, :]
        dw_part = tl.load(DW_partial_ptr + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        db_part = tl.load(DB_partial_ptr + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        acc_dw += tl.sum(dw_part, axis=0)
        acc_db += tl.sum(db_part, axis=0)

    tl.store(DW_ptr + cols, acc_dw, mask=mask_n)
    tl.store(DB_ptr + cols, acc_db, mask=mask_n)


@triton.jit
def _flash_causal_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    L_ptr,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q_ptr + off_z * stride_qz + off_h * stride_qh
    k_base = K_ptr + off_z * stride_kz + off_h * stride_kh
    v_base = V_ptr + off_z * stride_vz + off_h * stride_vh
    o_base = Out_ptr + off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_CTX

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e20
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        k_ptrs = (
            k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        )
        v_ptrs = (
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        )

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, k, allow_tf32=False) * sm_scale
        causal_mask = (
            (offs_m[:, None] >= offs_n[None, :])
            & mask_m[:, None]
            & mask_n[None, :]
        )
        qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_ij

    out = acc / l_i[:, None]
    l_ptrs = L_ptr + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=mask_m)

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, out, mask=mask_m[:, None])


@triton.jit
def _flash_causal_attn_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    DO_ptr,
    DQ_ptr,
    DK_ptr,
    DV_ptr,
    L_ptr,
    D_ptr,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    Z,
    H,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    base_offset = off_z * stride_qz + off_h * stride_qh
    q_base = Q_ptr + base_offset
    k_base = K_ptr + base_offset
    v_base = V_ptr + base_offset
    do_base = DO_ptr + base_offset
    dq_base = DQ_ptr + base_offset
    dk_base = DK_ptr + base_offset
    dv_base = DV_ptr + base_offset

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < N_CTX

    k_ptrs = k_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    v_ptrs = v_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_CTX

        q_ptrs = (
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        )
        do_ptrs = (
            do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        l_i = tl.load(
            L_ptr + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0
        ).to(tl.float32)
        d_i = tl.load(
            D_ptr + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0
        ).to(tl.float32)

        qk = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
        causal_mask = (
            (offs_m[:, None] >= offs_n[None, :])
            & mask_m[:, None]
            & mask_n[None, :]
        )
        p = tl.where(causal_mask, tl.exp(qk - l_i[:, None]), 0.0)

        dv += tl.dot(tl.trans(p), do, allow_tf32=False)
        dp = tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = p * (dp - d_i[:, None]) * sm_scale

        dk += tl.dot(tl.trans(ds), q, allow_tf32=False)
        dq_contrib = tl.dot(ds, k, allow_tf32=False)
        dq_ptrs = (
            dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        )
        tl.atomic_add(dq_ptrs, dq_contrib, mask=mask_m[:, None])

    dk_ptrs = (
        dk_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    dv_ptrs = (
        dv_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    tl.store(dk_ptrs, dk, mask=mask_n[:, None])
    tl.store(dv_ptrs, dv, mask=mask_n[:, None])


@triton.jit
def _cross_entropy_fwd_bwd_kernel(
    Logits_ptr,
    Targets_ptr,
    Loss_ptr,
    DLogits_ptr,
    stride_logits_m,
    N_ROWS,
    V,
    inv_n_rows,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V

    logits_ptrs = Logits_ptr + row * stride_logits_m + cols
    logits = tl.load(logits_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    target = tl.load(Targets_ptr + row)

    max_logit = tl.max(logits, axis=0)
    shifted = logits - max_logit
    exp_logits = tl.where(mask, tl.exp(shifted), 0.0)
    sum_exp = tl.sum(exp_logits, axis=0)
    lse = max_logit + tl.log(sum_exp)

    target_logit = tl.load(Logits_ptr + row * stride_logits_m + target).to(
        tl.float32
    )
    loss_i = lse - target_logit
    tl.store(Loss_ptr + row, loss_i)

    probs = exp_logits / sum_exp
    is_target = (cols == target).to(tl.float32)
    dlogits = (probs - is_target) * inv_n_rows
    tl.store(DLogits_ptr + row * stride_logits_m + cols, dlogits, mask=mask)


class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        x_contig = x.contiguous()
        orig_shape = x_contig.shape
        N = orig_shape[-1]
        M = x_contig.numel() // N
        x_2d = x_contig.view(M, N)

        if x.is_cuda and HAS_TRITON:
            y = torch.empty_like(x_2d)
            mean = torch.empty((M,), dtype=torch.float32, device=x.device)
            rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
            BLOCK_SIZE = triton.next_power_of_2(N)
            _layernorm_fwd_kernel[(M,)](
                x_2d,
                y,
                weight,
                bias,
                mean,
                rstd,
                x_2d.stride(0),
                N,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.save_for_backward(x_2d, weight, bias, mean, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.orig_shape = orig_shape
            ctx.eps = eps
            return y.view(orig_shape)

        mean = x_2d.mean(dim=-1, keepdim=True)
        var = ((x_2d - mean) ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = (x_2d - mean) * rstd
        y = x_hat * weight + bias
        ctx.save_for_backward(
            x_2d, weight, bias, mean.squeeze(-1), rstd.squeeze(-1)
        )
        ctx.orig_shape = orig_shape
        ctx.eps = eps
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight, bias, mean, rstd = ctx.saved_tensors
        M, N = x_2d.shape
        dy_2d = grad_output.contiguous().view(M, N)

        if x_2d.is_cuda and HAS_TRITON:
            dx = torch.empty_like(x_2d)
            dw_partial = torch.empty(
                (M, N), dtype=torch.float32, device=x_2d.device
            )
            db_partial = torch.empty(
                (M, N), dtype=torch.float32, device=x_2d.device
            )
            dw = torch.empty((N,), dtype=weight.dtype, device=x_2d.device)
            db = torch.empty((N,), dtype=bias.dtype, device=x_2d.device)

            _layernorm_bwd_dx_kernel[(M,)](
                dx,
                dy_2d,
                dw_partial,
                db_partial,
                x_2d,
                weight,
                mean,
                rstd,
                x_2d.stride(0),
                N,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
            )
            BLOCK_M = 128
            BLOCK_N = 64
            grid_reduce = (triton.cdiv(N, BLOCK_N),)
            _layernorm_bwd_dwdb_kernel[grid_reduce](
                dw_partial,
                db_partial,
                dw,
                db,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            return dx.view(ctx.orig_shape), dw, db, None

        mean_2d = mean.unsqueeze(-1)
        rstd_2d = rstd.unsqueeze(-1)
        xhat = (x_2d - mean_2d) * rstd_2d
        wdy = weight * dy_2d
        c1 = (xhat * wdy).mean(dim=-1, keepdim=True)
        c2 = wdy.mean(dim=-1, keepdim=True)
        dx = (wdy - (xhat * c1 + c2)) * rstd_2d
        dw = (dy_2d * xhat).sum(dim=0)
        db = dy_2d.sum(dim=0)
        return dx.view(ctx.orig_shape), dw, db, None


def triton_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return TritonLayerNormFunction.apply(x, weight, bias, eps)


class TritonCausalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        B, H, T, D = q.shape
        sm_scale = 1.0 / math.sqrt(D)

        if q.is_cuda and HAS_TRITON:
            out = torch.empty_like(q)
            L = torch.empty((B * H, T), dtype=torch.float32, device=q.device)
            BLOCK_M = 32
            BLOCK_N = 32
            BLOCK_DMODEL = triton.next_power_of_2(D)
            grid = (triton.cdiv(T, BLOCK_M), B * H)
            _flash_causal_attn_fwd_kernel[grid](
                q,
                k,
                v,
                out,
                L,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                B,
                H,
                T,
                sm_scale,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
            )
            ctx.save_for_backward(q, k, v, out, L)
            ctx.sm_scale = sm_scale
            ctx.BLOCK_M = BLOCK_M
            ctx.BLOCK_N = BLOCK_N
            ctx.BLOCK_DMODEL = BLOCK_DMODEL
            return out

        att = (q @ k.transpose(-2, -1)) * sm_scale
        causal_mask = torch.tril(
            torch.ones((T, T), device=q.device, dtype=torch.bool)
        )
        att = att.masked_fill(~causal_mask, float("-inf"))
        probs = F.softmax(att, dim=-1)
        out = probs @ v
        ctx.save_for_backward(q, k, v, probs)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        do = grad_output.contiguous()
        if do.is_cuda and HAS_TRITON:
            q, k, v, out, L = ctx.saved_tensors
            B, H, T, D = q.shape
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            delta = (
                (do.float() * out.float()).sum(dim=-1).reshape(B * H, T).contiguous()
            )
            grid = (triton.cdiv(T, ctx.BLOCK_N), B * H)
            _flash_causal_attn_bwd_kernel[grid](
                q,
                k,
                v,
                out,
                do,
                dq,
                dk,
                dv,
                L,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                B,
                H,
                T,
                ctx.sm_scale,
                BLOCK_M=ctx.BLOCK_M,
                BLOCK_N=ctx.BLOCK_N,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            )
            return dq.to(q.dtype), dk, dv

        q, k, v, probs = ctx.saved_tensors
        dv = probs.transpose(-2, -1) @ do
        dprobs = do @ v.transpose(-2, -1)
        ds = probs * (
            dprobs - (dprobs * probs).sum(dim=-1, keepdim=True)
        ) * ctx.sm_scale
        dq = ds @ k
        dk = ds.transpose(-2, -1) @ q
        return dq, dk, dv


def triton_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return TritonCausalAttentionFunction.apply(q, k, v)


class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits_2d = logits.contiguous().view(-1, logits.size(-1))
        targets_1d = targets.contiguous().view(-1)
        N_ROWS, V = logits_2d.shape

        if logits.is_cuda and HAS_TRITON:
            losses = torch.empty(
                (N_ROWS,), dtype=torch.float32, device=logits.device
            )
            dlogits = torch.empty_like(logits_2d, dtype=torch.float32)
            BLOCK_V = triton.next_power_of_2(V)
            _cross_entropy_fwd_bwd_kernel[(N_ROWS,)](
                logits_2d,
                targets_1d,
                losses,
                dlogits,
                logits_2d.stride(0),
                N_ROWS,
                V,
                1.0 / float(N_ROWS),
                BLOCK_V=BLOCK_V,
            )
            ctx.save_for_backward(dlogits.to(logits.dtype))
            ctx.orig_shape = logits.shape
            return losses.mean()

        log_probs = F.log_softmax(logits_2d, dim=-1)
        loss = F.nll_loss(log_probs, targets_1d, reduction="mean")
        probs = log_probs.exp()
        dlogits = probs.clone()
        dlogits[torch.arange(N_ROWS, device=logits.device), targets_1d] -= 1.0
        dlogits /= float(N_ROWS)
        ctx.save_for_backward(dlogits)
        ctx.orig_shape = logits.shape
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dlogits,) = ctx.saved_tensors
        return (dlogits * grad_output).view(ctx.orig_shape), None


def triton_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return TritonCrossEntropyFunction.apply(logits, targets)
