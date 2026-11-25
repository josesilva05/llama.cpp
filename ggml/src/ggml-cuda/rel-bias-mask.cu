#include "rel-bias-mask.cuh"
#include "convert.cuh"

#define CUDA_REL_BIAS_MASK_BLOCK_SIZE 256

// grid: (KV cell blocks, tokens per stream, streams); each thread reads the distance and the mask of one cell once
// and writes that cell for every head, so the writes of a head stay coalesced along the KV cells
template <typename TM, typename TD>
static __global__ void rel_bias_mask_kernel(
        const float * table, const int32_t * rel_pos, const TM * mask, TD * dst,
        const int64_t n_kv, const int64_t n_tok, const int64_t n_head, const int32_t n_dist,
        const int64_t st1, const int64_t st2,
        const int64_t sr1,
        const int64_t sm1, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    ggml_cuda_pdl_lc();

    const int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t s = blockIdx.z;
    const int64_t u = s*n_tok + t;

    ggml_cuda_pdl_sync();

    if (c >= n_kv) {
        return;
    }

    const int32_t d = min(max(rel_pos[u*sr1 + c], 0), n_dist - 1);
    const float   m = ggml_cuda_cast<float>(mask[t*sm1 + s*sm3 + c]);

    const float * tab = table + u*st2 + d;
    TD          * out = dst   + t*sd1 + s*sd3 + c;

    for (int64_t h = 0; h < n_head; ++h) {
        out[h*sd2] = ggml_cuda_cast<TD>(__ldg(tab + h*st1) + m);
    }
}

template <typename TM, typename TD>
static void rel_bias_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table   = dst->src[0];
    const ggml_tensor * rel_pos = dst->src[1];
    const ggml_tensor * mask    = dst->src[2];

    GGML_ASSERT(dst->nb[0] == sizeof(TD));

    const int64_t n_kv   = dst->ne[0];
    const int64_t n_tok  = dst->ne[1];
    const int64_t n_head = dst->ne[2];
    const int64_t n_strm = dst->ne[3];

    GGML_ASSERT(n_tok <= 65535 && n_strm <= 65535);

    const dim3 block_dims(CUDA_REL_BIAS_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims((n_kv + CUDA_REL_BIAS_MASK_BLOCK_SIZE - 1)/CUDA_REL_BIAS_MASK_BLOCK_SIZE, n_tok, n_strm);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(rel_bias_mask_kernel<TM, TD>, launch_params,
            (const float *) table->data, (const int32_t *) rel_pos->data, (const TM *) mask->data, (TD *) dst->data,
            n_kv, n_tok, n_head, (int32_t) table->ne[0],
            table->nb[1]/sizeof(float), table->nb[2]/sizeof(float),
            rel_pos->nb[1]/sizeof(int32_t),
            mask->nb[1]/sizeof(TM), mask->nb[3]/sizeof(TM),
            dst->nb[1]/sizeof(TD), dst->nb[2]/sizeof(TD), dst->nb[3]/sizeof(TD));
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type tm = dst->src[2]->type;
    const ggml_type td = dst->type;

    if (tm == GGML_TYPE_F16 && td == GGML_TYPE_F16) {
        rel_bias_mask_cuda<half, half>(ctx, dst);
    } else if (tm == GGML_TYPE_F16 && td == GGML_TYPE_F32) {
        rel_bias_mask_cuda<half, float>(ctx, dst);
    } else if (tm == GGML_TYPE_F32 && td == GGML_TYPE_F16) {
        rel_bias_mask_cuda<float, half>(ctx, dst);
    } else if (tm == GGML_TYPE_F32 && td == GGML_TYPE_F32) {
        rel_bias_mask_cuda<float, float>(ctx, dst);
    } else {
        GGML_ABORT("unsupported type");
    }
}
