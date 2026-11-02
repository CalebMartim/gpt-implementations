#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t stat = (call);                                          \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS Error at %s:%d: code %d\n", __FILE__,      \
                    __LINE__, (int)stat);                                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Model Hyperparameters (Fixed Architecture matching golden reference)
constexpr int V = 65;
constexpr int MAX_B = 64;
constexpr int MAX_T = 256;
constexpr int C = 384;
constexpr int H = 6;
constexpr int HS = 64;
constexpr int L = 6;
constexpr int MLP_C = 4 * C;
constexpr size_t NUM_PARAMS = 10788929ULL;

// Warp-level reduction primitives
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// 1. Token + Position Embedding Forward Kernel
__global__ void embedding_forward_kernel(
    float* __restrict__ out,
    const int32_t* __restrict__ idx,
    const float* __restrict__ wte,
    const float* __restrict__ wpe,
    int B,
    int T
) {
    int bt = blockIdx.x;
    int c = threadIdx.x;
    if (bt < B * T && c < C) {
        int t = bt % T;
        int token = idx[bt];
        out[bt * C + c] = wte[token * C + c] + wpe[t * C + c];
    }
}

// 2. Token + Position Embedding Backward Kernel
__global__ void embedding_backward_kernel(
    float* __restrict__ dwte,
    float* __restrict__ dwpe,
    const float* __restrict__ dout,
    const int32_t* __restrict__ idx,
    int B,
    int T
) {
    int bt = blockIdx.x;
    int c = threadIdx.x;
    if (bt < B * T && c < C) {
        int t = bt % T;
        int token = idx[bt];
        float g = dout[bt * C + c];
        atomicAdd(&dwte[token * C + c], g);
        atomicAdd(&dwpe[t * C + c], g);
    }
}

// 3. LayerNorm Forward Kernel (1 block per row bt, 128 threads = 4 warps)
__global__ void layernorm_forward_kernel(
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int BT
) {
    int bt = blockIdx.x;
    if (bt >= BT) return;

    const float* x = inp + bt * C;
    float* y = out + bt * C;

    __shared__ float s_sum[4];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        sum += x[c];
    }
    sum = warpReduceSum(sum);
    if (lane_id == 0) s_sum[warp_id] = sum;
    __syncthreads();

    if (tid == 0) {
        float total = s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3];
        s_mean = total / (float)C;
        if (mean_out) mean_out[bt] = s_mean;
    }
    __syncthreads();

    float m = s_mean;
    float vsum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float diff = x[c] - m;
        vsum += diff * diff;
    }
    vsum = warpReduceSum(vsum);
    if (lane_id == 0) s_sum[warp_id] = vsum;
    __syncthreads();

    if (tid == 0) {
        float total_var = (s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3]) / (float)C;
        s_rstd = rsqrtf(total_var + 1e-5f);
        if (rstd_out) rstd_out[bt] = s_rstd;
    }
    __syncthreads();

    float r = s_rstd;
    for (int c = tid; c < C; c += blockDim.x) {
        float norm = (x[c] - m) * r;
        y[c] = norm * weight[c] + bias[c];
    }
}

// 4. LayerNorm Backward Kernel
__global__ void layernorm_backward_kernel(
    float* __restrict__ dinp,
    float* __restrict__ dweight,
    float* __restrict__ dbias,
    const float* __restrict__ dout,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    int BT,
    bool accumulate_dinp
) {
    int bt = blockIdx.x;
    if (bt >= BT) return;

    const float* dy = dout + bt * C;
    const float* x = inp + bt * C;
    float* dx = dinp + bt * C;
    float m = mean[bt];
    float r = rstd[bt];

    __shared__ float s_sum1[4];
    __shared__ float s_sum2[4];
    __shared__ float s_c1;
    __shared__ float s_c2;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float norm = (x[c] - m) * r;
        float dnorm = dy[c] * weight[c];
        sum1 += dnorm;
        sum2 += dnorm * norm;
    }
    sum1 = warpReduceSum(sum1);
    sum2 = warpReduceSum(sum2);
    if (lane_id == 0) {
        s_sum1[warp_id] = sum1;
        s_sum2[warp_id] = sum2;
    }
    __syncthreads();

    if (tid == 0) {
        s_c1 = (s_sum1[0] + s_sum1[1] + s_sum1[2] + s_sum1[3]) / (float)C;
        s_c2 = (s_sum2[0] + s_sum2[1] + s_sum2[2] + s_sum2[3]) / (float)C;
    }
    __syncthreads();

    float c1 = s_c1;
    float c2 = s_c2;
    for (int c = tid; c < C; c += blockDim.x) {
        float norm = (x[c] - m) * r;
        float dnorm = dy[c] * weight[c];
        float g_x = r * (dnorm - c1 - norm * c2);
        if (accumulate_dinp) {
            dx[c] += g_x;
        } else {
            dx[c] = g_x;
        }
        atomicAdd(&dweight[c], dy[c] * norm);
        atomicAdd(&dbias[c], dy[c]);
    }
}

// 5. Bias Add Forward Kernel
__global__ void bias_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        out[idx] += bias[col];
    }
}

// 6. Bias Backward Kernel
__global__ void bias_backward_kernel(
    float* __restrict__ dbias,
    const float* __restrict__ dout,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; ++r) {
            sum += dout[r * cols + col];
        }
        dbias[col] += sum;
    }
}

// 7. Residual Stream Addition Kernel
__global__ void residual_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = a[idx] + b[idx];
    }
}

// 8. ReLU Forward Kernel
__global__ void relu_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = inp[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

// 9. ReLU Backward Kernel
__global__ void relu_backward_kernel(
    float* __restrict__ dinp,
    const float* __restrict__ dout,
    const float* __restrict__ inp,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dinp[idx] = inp[idx] > 0.0f ? dout[idx] : 0.0f;
    }
}

// 10. QKV Split + Transpose Forward: [B, T, 3*C] -> Q, K, V each [B, H, T, HS]
__global__ void qkv_split_transpose_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ qkv,
    int B,
    int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx < total) {
        int c = idx % C;
        int bt = idx / C;
        int t = bt % T;
        int b = bt / T;
        int h = c / HS;
        int d = c % HS;

        int out_idx = ((b * H + h) * T + t) * HS + d;
        int base_in = bt * (3 * C);
        q[out_idx] = qkv[base_in + c];
        k[out_idx] = qkv[base_in + C + c];
        v[out_idx] = qkv[base_in + 2 * C + c];
    }
}

// 11. QKV Transpose Backward: dQ, dK, dV each [B, H, T, HS] -> dQKV [B, T, 3*C]
__global__ void qkv_backward_transpose_kernel(
    float* __restrict__ dqkv,
    const float* __restrict__ dq,
    const float* __restrict__ dk,
    const float* __restrict__ dv,
    int B,
    int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx < total) {
        int c = idx % C;
        int bt = idx / C;
        int t = bt % T;
        int b = bt / T;
        int h = c / HS;
        int d = c % HS;

        int in_idx = ((b * H + h) * T + t) * HS + d;
        int base_out = bt * (3 * C);
        dqkv[base_out + c] = dq[in_idx];
        dqkv[base_out + C + c] = dk[in_idx];
        dqkv[base_out + 2 * C + c] = dv[in_idx];
    }
}

// 12. Causal Masked Scaled Softmax Forward Kernel (1 warp = 32 threads per row)
__global__ void causal_softmax_forward_kernel(
    float* __restrict__ probs,
    const float* __restrict__ scores,
    int BH,
    int T,
    float scale
) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row_idx >= BH * T) return;

    int t_row = row_idx % T;
    const float* s_row = scores + row_idx * T;
    float* p_row = probs + row_idx * T;

    float max_val = -INFINITY;
    for (int col = lane; col <= t_row; col += 32) {
        float val = s_row[col] * scale;
        max_val = fmaxf(max_val, val);
    }
    max_val = warpReduceMax(max_val);

    float sum_exp = 0.0f;
    for (int col = lane; col <= t_row; col += 32) {
        float val = expf(s_row[col] * scale - max_val);
        p_row[col] = val;
        sum_exp += val;
    }
    sum_exp = warpReduceSum(sum_exp);
    float inv_sum = 1.0f / sum_exp;

    for (int col = lane; col < T; col += 32) {
        if (col <= t_row) {
            p_row[col] *= inv_sum;
        } else {
            p_row[col] = 0.0f;
        }
    }
}

// 13. Causal Masked Scaled Softmax Backward Kernel (1 warp = 32 threads per row)
__global__ void causal_softmax_backward_kernel(
    float* __restrict__ dscores,
    const float* __restrict__ dprobs,
    const float* __restrict__ probs,
    int BH,
    int T,
    float scale
) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row_idx >= BH * T) return;

    int t_row = row_idx % T;
    const float* dp_row = dprobs + row_idx * T;
    const float* p_row = probs + row_idx * T;
    float* ds_row = dscores + row_idx * T;

    float dot = 0.0f;
    for (int col = lane; col <= t_row; col += 32) {
        dot += p_row[col] * dp_row[col];
    }
    dot = warpReduceSum(dot);

    for (int col = lane; col < T; col += 32) {
        if (col <= t_row) {
            float p = p_row[col];
            ds_row[col] = p * (dp_row[col] - dot) * scale;
        } else {
            ds_row[col] = 0.0f;
        }
    }
}

// 14. Attention Head Concat Transpose Forward/Backward Kernel
__global__ void attn_concat_transpose_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int B,
    int T,
    bool forward_dir
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx < total) {
        int c = idx % C;
        int bt = idx / C;
        int t = bt % T;
        int b = bt / T;
        int h = c / HS;
        int d = c % HS;

        int bhtd = ((b * H + h) * T + t) * HS + d;
        if (forward_dir) {
            out[idx] = inp[bhtd];
        } else {
            out[bhtd] = inp[idx];
        }
    }
}

// 15. Fused Cross-Entropy Forward + Backward Kernel (1 block = 32 threads per token bt)
__global__ void cross_entropy_forward_backward_kernel(
    float* __restrict__ losses,
    float* __restrict__ dlogits,
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int BT
) {
    int bt = blockIdx.x;
    int lane = threadIdx.x;
    if (bt >= BT) return;

    const float* l_row = logits + bt * V;
    float max_val = -INFINITY;
    for (int v = lane; v < V; v += 32) {
        max_val = fmaxf(max_val, l_row[v]);
    }
    max_val = warpReduceMax(max_val);

    float sum_exp = 0.0f;
    for (int v = lane; v < V; v += 32) {
        sum_exp += expf(l_row[v] - max_val);
    }
    sum_exp = warpReduceSum(sum_exp);

    int target = targets[bt];
    if (lane == 0) {
        losses[bt] = -(l_row[target] - max_val - logf(sum_exp));
    }

    if (dlogits != nullptr) {
        float* dl_row = dlogits + bt * V;
        float inv_bt = 1.0f / (float)BT;
        float inv_sum = 1.0f / sum_exp;
        for (int v = lane; v < V; v += 32) {
            float prob = expf(l_row[v] - max_val) * inv_sum;
            float ind = (v == target) ? 1.0f : 0.0f;
            dl_row[v] = (prob - ind) * inv_bt;
        }
    }
}

// 16. AdamW Optimizer Update Kernel (matches torch.optim.AdamW)
__global__ void adamw_update_kernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m_buf,
    float* __restrict__ v_buf,
    size_t num_params,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float beta1_correction,
    float beta2_correction
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        float p = params[idx];
        float g = grads[idx];

        p = p * (1.0f - lr * weight_decay);
        float m = beta1 * m_buf[idx] + (1.0f - beta1) * g;
        float v = beta2 * v_buf[idx] + (1.0f - beta2) * g * g;
        m_buf[idx] = m;
        v_buf[idx] = v;

        float m_hat = m / beta1_correction;
        float v_hat = v / beta2_correction;
        params[idx] = p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// 17. Greedy Argmax Kernel for last time step of each sequence in batch
__global__ void argmax_last_step_kernel(
    int32_t* __restrict__ next_tokens,
    const float* __restrict__ logits,
    int B,
    int cur_T
) {
    int b = blockIdx.x;
    if (b >= B) return;

    const float* row = logits + (b * cur_T + (cur_T - 1)) * V;
    float best_val = row[0];
    int best_idx = 0;
    for (int v = 1; v < V; ++v) {
        float val = row[v];
        if (val > best_val) {
            best_val = val;
            best_idx = v;
        }
    }
    next_tokens[b] = best_idx;
}

// Parameter Pointers into Contiguous Device Buffer
struct ParameterOffsets {
    float* wte;
    float* wpe;
    float* ln1_w[L];
    float* ln1_b[L];
    float* c_attn_w[L];
    float* c_proj_w[L];
    float* c_proj_b[L];
    float* ln2_w[L];
    float* ln2_b[L];
    float* mlp_fc_w[L];
    float* mlp_fc_b[L];
    float* mlp_proj_w[L];
    float* mlp_proj_b[L];
    float* ln_f_w;
    float* ln_f_b;
    float* lm_head_w;
    float* lm_head_b;
};

static void bind_parameter_offsets(ParameterOffsets& p, float* base) {
    size_t offset = 0;
    p.wte = base + offset; offset += V * C;
    p.wpe = base + offset; offset += MAX_T * C;
    for (int l = 0; l < L; ++l) {
        p.ln1_w[l] = base + offset; offset += C;
        p.ln1_b[l] = base + offset; offset += C;
        p.c_attn_w[l] = base + offset; offset += 3 * C * C;
        p.c_proj_w[l] = base + offset; offset += C * C;
        p.c_proj_b[l] = base + offset; offset += C;
        p.ln2_w[l] = base + offset; offset += C;
        p.ln2_b[l] = base + offset; offset += C;
        p.mlp_fc_w[l] = base + offset; offset += MLP_C * C;
        p.mlp_fc_b[l] = base + offset; offset += MLP_C;
        p.mlp_proj_w[l] = base + offset; offset += C * MLP_C;
        p.mlp_proj_b[l] = base + offset; offset += C;
    }
    p.ln_f_w = base + offset; offset += C;
    p.ln_f_b = base + offset; offset += C;
    p.lm_head_w = base + offset; offset += V * C;
    p.lm_head_b = base + offset; offset += V;
}

// Forward Activations and Backward Scratch Buffers
struct ActivationBuffers {
    float* emb;
    float* ln1_out[L];
    float* ln1_mean[L];
    float* ln1_rstd[L];
    float* qkv[L];
    float* q[L];
    float* k[L];
    float* v[L];
    float* att_scores[L];
    float* att_probs[L];
    float* att_v[L];
    float* att_concat[L];
    float* attn_proj[L];
    float* res1[L];
    float* ln2_out[L];
    float* ln2_mean[L];
    float* ln2_rstd[L];
    float* mlp_fc[L];
    float* mlp_relu[L];
    float* mlp_proj[L];
    float* res2[L];
    float* ln_f_out;
    float* ln_f_mean;
    float* ln_f_rstd;
    float* logits;
    float* losses;

    // Backward workspace buffers
    float* dlogits;
    float* d_res;
    float* d_ln_out;
    float* d_mlp_relu;
    float* d_mlp_fc;
    float* d_att_concat;
    float* d_att_v;
    float* d_att_probs;
    float* d_att_scores;
    float* dq;
    float* dk;
    float* dv;
    float* dqkv;
};

struct GPTCudaEngine {
    cublasHandle_t cublas;
    float* d_params;
    float* d_grads;
    float* d_m;
    float* d_v;
    ParameterOffsets params;
    ParameterOffsets grads;
    ActivationBuffers acts;
    int32_t* d_x;
    int32_t* d_y;
    float* h_losses;
};

static void allocate_engine(GPTCudaEngine& eng) {
    CUBLAS_CHECK(cublasCreate(&eng.cublas));
    CUBLAS_CHECK(cublasSetMathMode(eng.cublas, CUBLAS_PEDANTIC_MATH));

    CUDA_CHECK(cudaMalloc(&eng.d_params, NUM_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_grads, NUM_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_m, NUM_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_v, NUM_PARAMS * sizeof(float)));

    bind_parameter_offsets(eng.params, eng.d_params);
    bind_parameter_offsets(eng.grads, eng.d_grads);

    int max_BT = MAX_B * MAX_T;
    CUDA_CHECK(cudaMalloc(&eng.acts.emb, max_BT * C * sizeof(float)));
    for (int l = 0; l < L; ++l) {
        CUDA_CHECK(cudaMalloc(&eng.acts.ln1_out[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.ln1_mean[l], max_BT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.ln1_rstd[l], max_BT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.qkv[l], max_BT * 3 * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.q[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.k[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.v[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.att_scores[l], MAX_B * H * MAX_T * MAX_T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.att_probs[l], MAX_B * H * MAX_T * MAX_T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.att_v[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.att_concat[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.attn_proj[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.res1[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.ln2_out[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.ln2_mean[l], max_BT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.ln2_rstd[l], max_BT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.mlp_fc[l], max_BT * MLP_C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.mlp_relu[l], max_BT * MLP_C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.mlp_proj[l], max_BT * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.acts.res2[l], max_BT * C * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&eng.acts.ln_f_out, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.ln_f_mean, max_BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.ln_f_rstd, max_BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.logits, max_BT * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.losses, max_BT * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&eng.acts.dlogits, max_BT * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_res, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_ln_out, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_mlp_relu, max_BT * MLP_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_mlp_fc, max_BT * MLP_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_att_concat, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_att_v, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_att_probs, MAX_B * H * MAX_T * MAX_T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.d_att_scores, MAX_B * H * MAX_T * MAX_T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.dq, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.dk, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.dv, max_BT * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.acts.dqkv, max_BT * 3 * C * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&eng.d_x, max_BT * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&eng.d_y, max_BT * sizeof(int32_t)));
    eng.h_losses = (float*)malloc(max_BT * sizeof(float));
}

static void reset_weights_and_optimizer(GPTCudaEngine& eng, const float* h_golden_weights) {
    CUDA_CHECK(cudaMemcpy(eng.d_params, h_golden_weights, NUM_PARAMS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(eng.d_grads, 0, NUM_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMemset(eng.d_m, 0, NUM_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMemset(eng.d_v, 0, NUM_PARAMS * sizeof(float)));
}

// Helper for row-major Linear Forward: Y [M, N] = X [M, K] @ W^T [K, N] + bias [N]
static void linear_forward(
    cublasHandle_t handle,
    float* Y,
    const float* X,
    const float* W,
    const float* bias,
    int M,
    int N,
    int K
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, K,
        X, K,
        &beta,
        Y, N
    ));
    if (bias != nullptr) {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bias_add_kernel<<<blocks, threads>>>(Y, bias, M, N);
    }
}

// Helper for row-major Linear Backward:
// dX [M, K] = dY [M, N] @ W [N, K]
// dW [N, K] += dY^T [N, M] @ X [M, K]
// dbias [N] += sum_m dY [M, N]
static void linear_backward(
    cublasHandle_t handle,
    float* dX,
    float* dW,
    float* dbias,
    const float* dY,
    const float* X,
    const float* W,
    int M,
    int N,
    int K
) {
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        K, M, N,
        &alpha,
        W, K,
        dY, N,
        &beta_zero,
        dX, K
    ));

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        K, N, M,
        &alpha,
        X, K,
        dY, N,
        &beta_one,
        dW, K
    ));

    if (dbias != nullptr) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        bias_backward_kernel<<<blocks, threads>>>(dbias, dY, M, N);
    }
}

// Full Forward Pass for arbitrary (cur_B, cur_T)
static float gpt_forward(
    GPTCudaEngine& eng,
    const int32_t* d_idx,
    const int32_t* d_targets,
    int cur_B,
    int cur_T,
    bool compute_dlogits
) {
    int BT = cur_B * cur_T;
    int BH = cur_B * H;
    int threads = 256;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float scale = 1.0f / sqrtf((float)HS);

    embedding_forward_kernel<<<BT, C>>>(
        eng.acts.emb, d_idx, eng.params.wte, eng.params.wpe, cur_B, cur_T
    );

    for (int l = 0; l < L; ++l) {
        const float* x_in = (l == 0) ? eng.acts.emb : eng.acts.res2[l - 1];

        layernorm_forward_kernel<<<BT, 128>>>(
            eng.acts.ln1_out[l], eng.acts.ln1_mean[l], eng.acts.ln1_rstd[l],
            x_in, eng.params.ln1_w[l], eng.params.ln1_b[l], BT
        );

        linear_forward(
            eng.cublas, eng.acts.qkv[l], eng.acts.ln1_out[l],
            eng.params.c_attn_w[l], nullptr, BT, 3 * C, C
        );

        int blocks_btc = (BT * C + threads - 1) / threads;
        qkv_split_transpose_kernel<<<blocks_btc, threads>>>(
            eng.acts.q[l], eng.acts.k[l], eng.acts.v[l],
            eng.acts.qkv[l], cur_B, cur_T
        );

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            eng.cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            cur_T, cur_T, HS,
            &alpha,
            eng.acts.k[l], HS, (long long)(cur_T * HS),
            eng.acts.q[l], HS, (long long)(cur_T * HS),
            &beta,
            eng.acts.att_scores[l], cur_T, (long long)(cur_T * cur_T),
            BH
        ));

        dim3 sm_block(32, 8);
        int sm_grid = (BH * cur_T + 7) / 8;
        causal_softmax_forward_kernel<<<sm_grid, sm_block>>>(
            eng.acts.att_probs[l], eng.acts.att_scores[l], BH, cur_T, scale
        );

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            eng.cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            HS, cur_T, cur_T,
            &alpha,
            eng.acts.v[l], HS, (long long)(cur_T * HS),
            eng.acts.att_probs[l], cur_T, (long long)(cur_T * cur_T),
            &beta,
            eng.acts.att_v[l], HS, (long long)(cur_T * HS),
            BH
        ));

        attn_concat_transpose_kernel<<<blocks_btc, threads>>>(
            eng.acts.att_concat[l], eng.acts.att_v[l], cur_B, cur_T, true
        );

        linear_forward(
            eng.cublas, eng.acts.attn_proj[l], eng.acts.att_concat[l],
            eng.params.c_proj_w[l], eng.params.c_proj_b[l], BT, C, C
        );

        residual_add_kernel<<<blocks_btc, threads>>>(
            eng.acts.res1[l], x_in, eng.acts.attn_proj[l], BT * C
        );

        layernorm_forward_kernel<<<BT, 128>>>(
            eng.acts.ln2_out[l], eng.acts.ln2_mean[l], eng.acts.ln2_rstd[l],
            eng.acts.res1[l], eng.params.ln2_w[l], eng.params.ln2_b[l], BT
        );

        linear_forward(
            eng.cublas, eng.acts.mlp_fc[l], eng.acts.ln2_out[l],
            eng.params.mlp_fc_w[l], eng.params.mlp_fc_b[l], BT, MLP_C, C
        );

        int blocks_mlp = (BT * MLP_C + threads - 1) / threads;
        relu_forward_kernel<<<blocks_mlp, threads>>>(
            eng.acts.mlp_relu[l], eng.acts.mlp_fc[l], BT * MLP_C
        );

        linear_forward(
            eng.cublas, eng.acts.mlp_proj[l], eng.acts.mlp_relu[l],
            eng.params.mlp_proj_w[l], eng.params.mlp_proj_b[l], BT, C, MLP_C
        );

        residual_add_kernel<<<blocks_btc, threads>>>(
            eng.acts.res2[l], eng.acts.res1[l], eng.acts.mlp_proj[l], BT * C
        );
    }

    layernorm_forward_kernel<<<BT, 128>>>(
        eng.acts.ln_f_out, eng.acts.ln_f_mean, eng.acts.ln_f_rstd,
        eng.acts.res2[L - 1], eng.params.ln_f_w, eng.params.ln_f_b, BT
    );

    linear_forward(
        eng.cublas, eng.acts.logits, eng.acts.ln_f_out,
        eng.params.lm_head_w, eng.params.lm_head_b, BT, V, C
    );

    if (d_targets == nullptr) {
        return 0.0f;
    }

    cross_entropy_forward_backward_kernel<<<BT, 32>>>(
        eng.acts.losses,
        compute_dlogits ? eng.acts.dlogits : nullptr,
        eng.acts.logits,
        d_targets,
        BT
    );

    CUDA_CHECK(cudaMemcpy(eng.h_losses, eng.acts.losses, BT * sizeof(float), cudaMemcpyDeviceToHost));
    double total_loss = 0.0;
    for (int i = 0; i < BT; ++i) {
        total_loss += (double)eng.h_losses[i];
    }
    return (float)(total_loss / (double)BT);
}

// Full Backward Pass
static void gpt_backward(GPTCudaEngine& eng, const int32_t* d_idx, int cur_B, int cur_T) {
    int BT = cur_B * cur_T;
    int BH = cur_B * H;
    int threads = 256;
    int blocks_btc = (BT * C + threads - 1) / threads;
    int blocks_mlp = (BT * MLP_C + threads - 1) / threads;
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float scale = 1.0f / sqrtf((float)HS);

    CUDA_CHECK(cudaMemset(eng.d_grads, 0, NUM_PARAMS * sizeof(float)));

    // Backward through lm_head -> d_ln_out = d(ln_f_out)
    linear_backward(
        eng.cublas,
        eng.acts.d_ln_out,
        eng.grads.lm_head_w,
        eng.grads.lm_head_b,
        eng.acts.dlogits,
        eng.acts.ln_f_out,
        eng.params.lm_head_w,
        BT, V, C
    );

    // Backward through ln_f -> d_res = d(res2[L-1])
    layernorm_backward_kernel<<<BT, 128>>>(
        eng.acts.d_res,
        eng.grads.ln_f_w,
        eng.grads.ln_f_b,
        eng.acts.d_ln_out,
        eng.acts.res2[L - 1],
        eng.params.ln_f_w,
        eng.acts.ln_f_mean,
        eng.acts.ln_f_rstd,
        BT,
        false
    );

    for (int l = L - 1; l >= 0; --l) {
        // MLP backward: d_res currently holds d(res2[l])
        linear_backward(
            eng.cublas,
            eng.acts.d_mlp_relu,
            eng.grads.mlp_proj_w[l],
            eng.grads.mlp_proj_b[l],
            eng.acts.d_res,
            eng.acts.mlp_relu[l],
            eng.params.mlp_proj_w[l],
            BT, C, MLP_C
        );

        relu_backward_kernel<<<blocks_mlp, threads>>>(
            eng.acts.d_mlp_fc, eng.acts.d_mlp_relu, eng.acts.mlp_fc[l], BT * MLP_C
        );

        linear_backward(
            eng.cublas,
            eng.acts.d_ln_out,
            eng.grads.mlp_fc_w[l],
            eng.grads.mlp_fc_b[l],
            eng.acts.d_mlp_fc,
            eng.acts.ln2_out[l],
            eng.params.mlp_fc_w[l],
            BT, MLP_C, C
        );

        // Accumulate into d_res to form d(res1[l])
        layernorm_backward_kernel<<<BT, 128>>>(
            eng.acts.d_res,
            eng.grads.ln2_w[l],
            eng.grads.ln2_b[l],
            eng.acts.d_ln_out,
            eng.acts.res1[l],
            eng.params.ln2_w[l],
            eng.acts.ln2_mean[l],
            eng.acts.ln2_rstd[l],
            BT,
            true
        );

        // Attention backward: d_res holds d(res1[l])
        linear_backward(
            eng.cublas,
            eng.acts.d_att_concat,
            eng.grads.c_proj_w[l],
            eng.grads.c_proj_b[l],
            eng.acts.d_res,
            eng.acts.att_concat[l],
            eng.params.c_proj_w[l],
            BT, C, C
        );

        attn_concat_transpose_kernel<<<blocks_btc, threads>>>(
            eng.acts.d_att_v, eng.acts.d_att_concat, cur_B, cur_T, false
        );

        // dP = d_att_v @ V^T
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            eng.cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            cur_T, cur_T, HS,
            &alpha,
            eng.acts.v[l], HS, (long long)(cur_T * HS),
            eng.acts.d_att_v, HS, (long long)(cur_T * HS),
            &beta_zero,
            eng.acts.d_att_probs, cur_T, (long long)(cur_T * cur_T),
            BH
        ));

        // dV = P^T @ d_att_v
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            eng.cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            HS, cur_T, cur_T,
            &alpha,
            eng.acts.d_att_v, HS, (long long)(cur_T * HS),
            eng.acts.att_probs[l], cur_T, (long long)(cur_T * cur_T),
            &beta_zero,
            eng.acts.dv, HS, (long long)(cur_T * HS),
            BH
        ));

        dim3 sm_block(32, 8);
        int sm_grid = (BH * cur_T + 7) / 8;
        causal_softmax_backward_kernel<<<sm_grid, sm_block>>>(
            eng.acts.d_att_scores,
            eng.acts.d_att_probs,
            eng.acts.att_probs[l],
            BH,
            cur_T,
            scale
        );

        // dQ = dS @ K
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            eng.cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            HS, cur_T, cur_T,
            &alpha,
            eng.acts.k[l], HS, (long long)(cur_T * HS),
            eng.acts.d_att_scores, cur_T, (long long)(cur_T * cur_T),
            &beta_zero,
            eng.acts.dq, HS, (long long)(cur_T * HS),
            BH
        ));

        // dK = dS^T @ Q
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            eng.cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            HS, cur_T, cur_T,
            &alpha,
            eng.acts.q[l], HS, (long long)(cur_T * HS),
            eng.acts.d_att_scores, cur_T, (long long)(cur_T * cur_T),
            &beta_zero,
            eng.acts.dk, HS, (long long)(cur_T * HS),
            BH
        ));

        qkv_backward_transpose_kernel<<<blocks_btc, threads>>>(
            eng.acts.dqkv, eng.acts.dq, eng.acts.dk, eng.acts.dv, cur_B, cur_T
        );

        linear_backward(
            eng.cublas,
            eng.acts.d_ln_out,
            eng.grads.c_attn_w[l],
            nullptr,
            eng.acts.dqkv,
            eng.acts.ln1_out[l],
            eng.params.c_attn_w[l],
            BT, 3 * C, C
        );

        const float* x_in = (l == 0) ? eng.acts.emb : eng.acts.res2[l - 1];
        layernorm_backward_kernel<<<BT, 128>>>(
            eng.acts.d_res,
            eng.grads.ln1_w[l],
            eng.grads.ln1_b[l],
            eng.acts.d_ln_out,
            x_in,
            eng.params.ln1_w[l],
            eng.acts.ln1_mean[l],
            eng.acts.ln1_rstd[l],
            BT,
            true
        );
    }

    embedding_backward_kernel<<<BT, C>>>(
        eng.grads.wte, eng.grads.wpe, eng.acts.d_res, d_idx, cur_B, cur_T
    );
}

static void gpt_adamw_step(GPTCudaEngine& eng, int step_num) {
    const float lr = 3e-4f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    float beta1_corr = 1.0f - powf(beta1, (float)step_num);
    float beta2_corr = 1.0f - powf(beta2, (float)step_num);

    int threads = 256;
    int blocks = (int)((NUM_PARAMS + threads - 1) / threads);
    adamw_update_kernel<<<blocks, threads>>>(
        eng.d_params,
        eng.d_grads,
        eng.d_m,
        eng.d_v,
        NUM_PARAMS,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        beta1_corr,
        beta2_corr
    );
}

// Autoregressive greedy token generation without KV-cache (exact parity with PyTorch)
static void generate_tokens(GPTCudaEngine& eng, int gen_B, int max_new_tokens) {
    std::vector<int32_t> seq(gen_B * (1 + max_new_tokens), 0);
    std::vector<int32_t> window(gen_B * MAX_T, 0);
    std::vector<int32_t> h_next(gen_B, 0);
    int32_t* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next, gen_B * sizeof(int32_t)));

    int cur_len = 1;
    for (int step = 0; step < max_new_tokens; ++step) {
        int cur_T = (cur_len < MAX_T) ? cur_len : MAX_T;
        int start_pos = cur_len - cur_T;
        for (int b = 0; b < gen_B; ++b) {
            for (int t = 0; t < cur_T; ++t) {
                window[b * cur_T + t] = seq[b * (1 + max_new_tokens) + start_pos + t];
            }
        }
        CUDA_CHECK(cudaMemcpy(eng.d_x, window.data(), gen_B * cur_T * sizeof(int32_t), cudaMemcpyHostToDevice));
        gpt_forward(eng, eng.d_x, nullptr, gen_B, cur_T, false);
        argmax_last_step_kernel<<<gen_B, 1>>>(d_next, eng.acts.logits, gen_B, cur_T);
        CUDA_CHECK(cudaMemcpy(h_next.data(), d_next, gen_B * sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (int b = 0; b < gen_B; ++b) {
            seq[b * (1 + max_new_tokens) + cur_len] = h_next[b];
        }
        cur_len++;
    }
    CUDA_CHECK(cudaFree(d_next));
}

// === BENCHMARK HARNESS ===

static double wall_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float evaluate_split(
    GPTCudaEngine& eng,
    const int32_t* split_x,
    const int32_t* split_y,
    int eval_iters,
    int avail_batches
) {
    int BT = MAX_B * MAX_T;
    double sum_loss = 0.0;
    for (int k = 0; k < eval_iters; ++k) {
        int idx = k % avail_batches;
        CUDA_CHECK(cudaMemcpy(eng.d_x, split_x + (size_t)idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.d_y, split_y + (size_t)idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
        float loss = gpt_forward(eng, eng.d_x, eng.d_y, MAX_B, MAX_T, false);
        sum_loss += (double)loss;
    }
    return (float)(sum_loss / (double)eval_iters);
}

struct LossPoint {
    int step;
    float train_loss;
    float val_loss;
};

int main(int argc, char** argv) {
    if (argc < 10) {
        fprintf(stderr,
                "Usage: %s <golden_dir> <output_json> <max_iters> <warmup_iters> "
                "<eval_interval> <eval_iters> <gen_tokens> <expected_loss0> <expected_loss1> "
                "[num_golden_steps] [num_golden_eval_iters]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    std::string golden_dir = argv[1];
    std::string output_json = argv[2];
    int max_iters = atoi(argv[3]);
    int warmup_iters = atoi(argv[4]);
    int eval_interval = atoi(argv[5]);
    int eval_iters = atoi(argv[6]);
    int gen_tokens = atoi(argv[7]);
    float expected_loss0 = strtof(argv[8], nullptr);
    float expected_loss1 = strtof(argv[9], nullptr);
    int golden_steps = (argc >= 11) ? atoi(argv[10]) : max_iters;
    int golden_eval_iters = (argc >= 12) ? atoi(argv[11]) : eval_iters;

    // Load golden weights
    std::string weights_file = golden_dir + "/golden_weights.bin";
    FILE* fw = fopen(weights_file.c_str(), "rb");
    if (!fw) {
        fprintf(stderr, "Failed to open %s\n", weights_file.c_str());
        return EXIT_FAILURE;
    }
    std::vector<float> h_weights(NUM_PARAMS);
    size_t n_read_w = fread(h_weights.data(), sizeof(float), NUM_PARAMS, fw);
    fclose(fw);
    if (n_read_w != NUM_PARAMS) {
        fprintf(stderr, "Expected %zu floats in %s, got %zu\n", NUM_PARAMS, weights_file.c_str(), n_read_w);
        return EXIT_FAILURE;
    }

    // Load golden batches
    std::string batches_file = golden_dir + "/golden_batches.bin";
    FILE* fb = fopen(batches_file.c_str(), "rb");
    if (!fb) {
        fprintf(stderr, "Failed to open %s\n", batches_file.c_str());
        return EXIT_FAILURE;
    }
    size_t BT = (size_t)(MAX_B * MAX_T);
    size_t total_ints = (2 * (size_t)golden_steps + 4 * (size_t)golden_eval_iters) * BT;
    std::vector<int32_t> h_batches(total_ints);
    size_t n_read_b = fread(h_batches.data(), sizeof(int32_t), total_ints, fb);
    fclose(fb);
    if (n_read_b != total_ints) {
        fprintf(stderr, "Expected %zu ints in %s, got %zu\n", total_ints, batches_file.c_str(), n_read_b);
        return EXIT_FAILURE;
    }

    const int32_t* train_x = h_batches.data();
    const int32_t* train_y = train_x + (size_t)golden_steps * BT;
    const int32_t* eval_train_x = train_y + (size_t)golden_steps * BT;
    const int32_t* eval_train_y = eval_train_x + (size_t)golden_eval_iters * BT;
    const int32_t* val_x = eval_train_y + (size_t)golden_eval_iters * BT;
    const int32_t* val_y = val_x + (size_t)golden_eval_iters * BT;

    GPTCudaEngine eng;
    allocate_engine(eng);

    // 1. Sanity Check Gate
    reset_weights_and_optimizer(eng, h_weights.data());
    CUDA_CHECK(cudaMemcpy(eng.d_x, train_x, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(eng.d_y, train_y, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
    float loss_0 = gpt_forward(eng, eng.d_x, eng.d_y, MAX_B, MAX_T, true);
    gpt_backward(eng, eng.d_x, MAX_B, MAX_T);
    gpt_adamw_step(eng, 1);

    int next_idx = (golden_steps > 1) ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(eng.d_x, train_x + (size_t)next_idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(eng.d_y, train_y + (size_t)next_idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
    float loss_1 = gpt_forward(eng, eng.d_x, eng.d_y, MAX_B, MAX_T, false);

    float diff_0 = fabsf(loss_0 - expected_loss0);
    float diff_1 = fabsf(loss_1 - expected_loss1);
    if (diff_0 >= 1e-4f || diff_1 >= 1e-4f) {
        fprintf(stderr,
                "Sanity check failed! step0=%f (expected %f, diff=%e), step1=%f (expected %f, diff=%e)\n",
                loss_0, expected_loss0, diff_0, loss_1, expected_loss1, diff_1);
        return EXIT_FAILURE;
    }

    // 2. Warm-up Phase
    reset_weights_and_optimizer(eng, h_weights.data());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_warm_start = wall_time_sec();
    for (int i = 0; i < warmup_iters; ++i) {
        int idx = i % golden_steps;
        CUDA_CHECK(cudaMemcpy(eng.d_x, train_x + (size_t)idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.d_y, train_y + (size_t)idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
        gpt_forward(eng, eng.d_x, eng.d_y, MAX_B, MAX_T, true);
        gpt_backward(eng, eng.d_x, MAX_B, MAX_T);
        gpt_adamw_step(eng, i + 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double warmup_time_s = wall_time_sec() - t_warm_start;

    // 3. Timed Training Phase (starts cleanly from theta_0)
    reset_weights_and_optimizer(eng, h_weights.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<LossPoint> loss_history;
    double train_wall_time_s = 0.0;
    double eval_wall_time_s = 0.0;

    double t_eval0 = wall_time_sec();
    float tr_loss0 = evaluate_split(eng, eval_train_x, eval_train_y, eval_iters, golden_eval_iters);
    float va_loss0 = evaluate_split(eng, val_x, val_y, eval_iters, golden_eval_iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    eval_wall_time_s += wall_time_sec() - t_eval0;
    loss_history.push_back({0, tr_loss0, va_loss0});

    double t_seg_start = wall_time_sec();
    for (int step = 0; step < max_iters; ++step) {
        int idx = step % golden_steps;
        CUDA_CHECK(cudaMemcpy(eng.d_x, train_x + (size_t)idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.d_y, train_y + (size_t)idx * BT, BT * sizeof(int32_t), cudaMemcpyHostToDevice));
        gpt_forward(eng, eng.d_x, eng.d_y, MAX_B, MAX_T, true);
        gpt_backward(eng, eng.d_x, MAX_B, MAX_T);
        gpt_adamw_step(eng, step + 1);

        bool is_eval = ((step > 0 && step % eval_interval == 0) || step == max_iters - 1);
        if (is_eval) {
            CUDA_CHECK(cudaDeviceSynchronize());
            train_wall_time_s += wall_time_sec() - t_seg_start;

            double t_ev = wall_time_sec();
            float tr_l = evaluate_split(eng, eval_train_x, eval_train_y, eval_iters, golden_eval_iters);
            float va_l = evaluate_split(eng, val_x, val_y, eval_iters, golden_eval_iters);
            CUDA_CHECK(cudaDeviceSynchronize());
            eval_wall_time_s += wall_time_sec() - t_ev;

            loss_history.push_back({step, tr_l, va_l});
            t_seg_start = wall_time_sec();
        }
    }

    // 4. Timed Inference Phase
    generate_tokens(eng, 1, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_b1 = wall_time_sec();
    generate_tokens(eng, 1, gen_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());
    double b1_time_s = wall_time_sec() - t_b1;
    double b1_tps = (double)gen_tokens / fmax(b1_time_s, 1e-9);
    double b1_lat_ms = (b1_time_s * 1000.0) / fmax((double)gen_tokens, 1.0);

    generate_tokens(eng, MAX_B, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_b64 = wall_time_sec();
    generate_tokens(eng, MAX_B, gen_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());
    double b64_time_s = wall_time_sec() - t_b64;
    double b64_tps = (double)(MAX_B * gen_tokens) / fmax(b64_time_s, 1e-9);

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    double peak_vram_mb = (double)(total_bytes - free_bytes) / (1024.0 * 1024.0);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    long long total_train_tokens = (long long)max_iters * MAX_B * MAX_T;
    double train_tps = (double)total_train_tokens / fmax(train_wall_time_s, 1e-9);
    float final_train_loss = loss_history.back().train_loss;
    float final_val_loss = loss_history.back().val_loss;

    FILE* fj = fopen(output_json.c_str(), "w");
    if (!fj) {
        fprintf(stderr, "Failed to write %s\n", output_json.c_str());
        return EXIT_FAILURE;
    }

    fprintf(fj, "{\n");
    fprintf(fj, "  \"implementation\": \"03_cuda_cpp\",\n");
    fprintf(fj, "  \"language_paradigm\": \"Native C++/CUDA + cuBLAS\",\n");
    fprintf(fj, "  \"gpu_name\": \"%s\",\n", prop.name);
    fprintf(fj, "  \"device\": \"cuda:0\",\n");
    fprintf(fj, "  \"train_time_s\": %.6f,\n", train_wall_time_s);
    fprintf(fj, "  \"train_tokens_per_sec\": %.2f,\n", train_tps);
    fprintf(fj, "  \"eval_time_s\": %.6f,\n", eval_wall_time_s);
    fprintf(fj, "  \"warmup_time_s\": %.6f,\n", warmup_time_s);
    fprintf(fj, "  \"infer_b1_time_s\": %.6f,\n", b1_time_s);
    fprintf(fj, "  \"infer_b1_tokens_per_sec\": %.2f,\n", b1_tps);
    fprintf(fj, "  \"infer_b1_latency_ms_per_token\": %.4f,\n", b1_lat_ms);
    fprintf(fj, "  \"infer_b64_time_s\": %.6f,\n", b64_time_s);
    fprintf(fj, "  \"infer_b64_tokens_per_sec\": %.2f,\n", b64_tps);
    fprintf(fj, "  \"peak_vram_mb\": %.2f,\n", peak_vram_mb);
    fprintf(fj, "  \"final_train_loss\": %.6f,\n", final_train_loss);
    fprintf(fj, "  \"final_val_loss\": %.6f,\n", final_val_loss);
    fprintf(fj, "  \"sanity_check\": {\n");
    fprintf(fj, "    \"loss_step_0\": %.6f,\n", loss_0);
    fprintf(fj, "    \"golden_loss_step_0\": %.6f,\n", expected_loss0);
    fprintf(fj, "    \"loss_step_0_diff\": %.8f,\n", diff_0);
    fprintf(fj, "    \"loss_step_1\": %.6f,\n", loss_1);
    fprintf(fj, "    \"golden_loss_step_1\": %.6f,\n", expected_loss1);
    fprintf(fj, "    \"loss_step_1_diff\": %.8f,\n", diff_1);
    fprintf(fj, "    \"passed\": true\n");
    fprintf(fj, "  },\n");
    fprintf(fj, "  \"training\": {\n");
    fprintf(fj, "    \"max_iters\": %d,\n", max_iters);
    fprintf(fj, "    \"warmup_iters\": %d,\n", warmup_iters);
    fprintf(fj, "    \"warmup_time_s\": %.6f,\n", warmup_time_s);
    fprintf(fj, "    \"wall_time_s\": %.6f,\n", train_wall_time_s);
    fprintf(fj, "    \"eval_wall_time_s\": %.6f,\n", eval_wall_time_s);
    fprintf(fj, "    \"total_tokens\": %lld,\n", total_train_tokens);
    fprintf(fj, "    \"throughput_tok_s\": %.2f,\n", train_tps);
    fprintf(fj, "    \"final_train_loss\": %.6f,\n", final_train_loss);
    fprintf(fj, "    \"final_val_loss\": %.6f\n", final_val_loss);
    fprintf(fj, "  },\n");
    fprintf(fj, "  \"inference\": {\n");
    fprintf(fj, "    \"gen_tokens\": %d,\n", gen_tokens);
    fprintf(fj, "    \"batch_1\": {\n");
    fprintf(fj, "      \"wall_time_s\": %.6f,\n", b1_time_s);
    fprintf(fj, "      \"throughput_tok_s\": %.2f,\n", b1_tps);
    fprintf(fj, "      \"latency_ms_per_token\": %.4f\n", b1_lat_ms);
    fprintf(fj, "    },\n");
    fprintf(fj, "    \"batch_64\": {\n");
    fprintf(fj, "      \"wall_time_s\": %.6f,\n", b64_time_s);
    fprintf(fj, "      \"throughput_tok_s\": %.2f\n", b64_tps);
    fprintf(fj, "    }\n");
    fprintf(fj, "  },\n");
    fprintf(fj, "  \"memory\": {\n");
    fprintf(fj, "    \"peak_vram_mb\": %.2f\n", peak_vram_mb);
    fprintf(fj, "  },\n");
    fprintf(fj, "  \"loss_history\": [\n");
    for (size_t i = 0; i < loss_history.size(); ++i) {
        const auto& pt = loss_history[i];
        fprintf(fj, "    {\"step\": %d, \"train_loss\": %.6f, \"val_loss\": %.6f}%s\n",
                pt.step, pt.train_loss, pt.val_loss,
                (i + 1 < loss_history.size()) ? "," : "");
    }
    fprintf(fj, "  ]\n");
    fprintf(fj, "}\n");
    fclose(fj);

    return EXIT_SUCCESS;
}
