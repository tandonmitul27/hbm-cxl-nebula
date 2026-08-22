"""GPU compute model: explicit roofline, datasheet numbers, one place.

Scope (per the professor): GPU compute is rough-but-explainable; the memory
system is the object of study.  This module makes the roughness explicit:

    T_layer = max( FLOPs / FLOPS_peak ,  bytes_touched / BW_hbm )

* Decode at batch B reads every touched weight once and does ~2*B FLOPs per
  weight, so arithmetic intensity ~ B FLOPs/byte.  The H100 fp16 ridge point
  is 989.4e12 / 3350e9 = 295 FLOPs/byte -- decode at any measurable batch is
  BANDWIDTH-bound and T_layer collapses to bytes/BW (the max() keeps the
  model honest anyway).
* Prefill processes B*P tokens per weight read: intensity ~ 2*B*P, far past
  the ridge -- COMPUTE-bound, T_layer = FLOPs/FLOPS_peak.
* FLOPs counted are the parameter GEMMs (2 * params * tokens); the
  attention-score quadratic term is omitted (second-order at our prompt
  lengths) -- stated limitation.
* Datasheet FLOPS are DENSE tensor-core figures (sparsity figures excluded).

Anchor check available on this box: measured RTX 4070 decode layer times from
Phase 1 sit within ~2x of bytes/BW_4070, the right ballpark for a roofline
with no launch-overhead model (documented in data/README.md).

Sources: NVIDIA H100 datasheet (SXM5), NVIDIA H200 datasheet, NVIDIA
Blackwell architecture brief.  fp8 halves bytes (dtype_bytes=1 in
address_map) and doubles peak FLOPS.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    fp16_tflops: float      # dense tensor-core, datasheet
    fp8_tflops: float
    hbm_gbps: float         # aggregate datasheet bandwidth
    hbm_gib: float
    stacks: int
    hbm_gen: str


GPUS = {
    "H100": GPU("H100_SXM", 989.4, 1978.9, 3350.0, 80, 5, "HBM3"),
    "H200": GPU("H200_SXM", 989.4, 1978.9, 4800.0, 141, 6, "HBM3e"),
    "B200": GPU("B200", 2250.0, 4500.0, 8000.0, 192, 8, "HBM3e"),
}


def peak_flops(gpu: GPU, dtype_bytes: int) -> float:
    if dtype_bytes == 2:
        return gpu.fp16_tflops * 1e12
    if dtype_bytes == 1:
        return gpu.fp8_tflops * 1e12
    raise ValueError(f"no datasheet FLOPS for dtype_bytes={dtype_bytes}")


def t_layer(bytes_touched: float, params_touched: float, tokens: int,
            gpu: GPU, dtype_bytes: int, hbm_bps: float | None = None,
            t0_s: float = 0.0, extra_flops: float = 0.0) -> float:
    """Roofline time for one layer barrier.

    bytes_touched  weights read from HBM for this layer
    params_touched parameter count behind those bytes
    tokens         tokens processed at this barrier (decode: B; prefill: B*P)
    hbm_bps        override for calibrated bandwidth; default datasheet
    t0_s           per-barrier launch/dispatch/sync overhead, OUTSIDE the
                   max(): it is serial host-side work, hidden by neither
                   bandwidth nor compute.  BRACKETED [5, 40] us, not fitted:
                   a CUDA kernel launch costs ~3-10 us and a decode layer is
                   ~10-15 kernels; CUDA-graph capture (vLLM, TensorRT-LLM)
                   amortises that to ~5-20 us/layer.  The Phase-1 RTX 4070
                   timings are only an upper-bound consistency check -- the
                   transformers Python expert loop inflates them ~5x
                   (data/README.md), so fitting t0 to them would bake a
                   framework artifact in as if it were hardware.
    extra_flops    FLOPs beyond the parameter GEMMs (attention quadratic)
    """
    bw = hbm_bps if hbm_bps is not None else gpu.hbm_gbps * 1e9
    t_mem = bytes_touched / bw
    t_cmp = ((2.0 * params_touched * tokens + extra_flops)
             / peak_flops(gpu, dtype_bytes))
    return t0_s + max(t_mem, t_cmp)


def attn_quad_flops(att: dict, batch: int, prompt: int) -> float:
    """Attention-score FLOPs for one PREFILL layer barrier: QK^T plus A*V,
    causal-masked (half the full P^2 product).

        scores  ~ P^2/2 * n_heads * qk_dim * 2 FLOPs
        A*V     ~ P^2/2 * n_heads * v_dim  * 2 FLOPs

    MLA scores run at the decompressed head dims (qk_nope + qk_rope / v_head);
    GQA does not shrink this term -- scores are per query head.

    Decode's quadratic term is deliberately NOT modelled as compute: per step
    it is ~2*B*(P+step)*d FLOPs, orders below the ridge, and its real cost --
    reading the KV cache -- is already charged as memory traffic by
    kv_layer_fn.
    """
    if att.get("mla"):
        qk = att["qk_nope"] + att["qk_rope"]
        v = att["v_head"]
    else:
        qk = v = att["head_dim"]
    return float(batch) * prompt * prompt * att["n_heads"] * (qk + v)
