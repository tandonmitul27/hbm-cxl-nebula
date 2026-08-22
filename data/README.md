# MoE routing logs

17,136,896 routing decisions extracted from five models on an RTX 4070.
Each record answers one question: **at this token, at this layer, which experts
did the router select?**

Nothing here is CXL-specific. This is raw observation; placement policy and
address mapping happen in `analytical/` and `mapping/`.

These logs are the analytical model's only workload input — `analytical/trace_gen.py`
walks one of them per operating point. `make routing` verifies all 20 against
the geometry they were recorded from.

## Layout

```
data/
  routing/<model>/b<batch>_p<prompt>_g<gen>.parquet   the routing log
  routing/index.json                                   which runs exist
  check_routing.py                                     integrity checks
```

Model **geometry** is not duplicated here — it lives once in
`mapping/geometry/<model>.json` and is read through `mapping/address_map.py`,
so the two cannot drift. `index.json` carries only what identifies a run:
batch, prompt and generation length, row count, and the file it names.

One parquet per `(model, batch, prompt_len, gen_len)`. The filename encodes the
configuration: `b64_p256_g32` is batch 64, a 256-token prompt, 32 generated
tokens.

## What is in the set

| model | N experts | top-k | MoE layers | expert size (fp16) | fetch @60 GB/s | records |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 64 | 8 | 16 / 16 | 12.0 MiB | 0.21 ms | 6,963,200 |
| DeepSeek-V2-Lite | 64 | 6 | 26 / 27 | 16.5 MiB | 0.29 ms | 5,081,856 |
| Phi-3.5-MoE | 16 | 2 | 32 / 32 | 150.0 MiB | 2.62 ms | 1,566,720 |
| Mixtral-8x7B | 8 | 2 | 32 / 32 | 336.0 MiB | 5.87 ms | 1,566,720 |
| Qwen2.5-3B (dense) | 1 | 1 | 36 / 36 | 129.0 MiB | 2.25 ms | 1,958,400 |

Each run sweeps batch 1 / 4 / 16 / 64. The four MoE models give a 28x spread in
expert size, which is the axis the fine- vs coarse-grained question turns on.

Sequence lengths differ by model and are recorded per run: the fine-grained pair
used 512-token prompts, the two coarse models 256 (they are 78-87 GiB and had to
stream from NVMe, at ~50 s per decode step).

**Qwen2.5-3B is the dense contrast case**, recorded as "one expert that every
token selects at every layer". The log is synthetic but exactly true, so the
model consumes it with no special-casing: the address trace it produces is the
every-token-touches-all-weights pattern that makes tiering useless.

**DeepSeek's layer 0 is dense**, so it contributes 26 MoE layers out of 27. The
`moe_layers` list in the metadata is authoritative — do not assume `range(n)`.

## Schema — one row per (token, layer, slot)

| column | type | meaning |
|---|---|---|
| `phase` | int8 | **0 = prefill, 1 = decode** |
| `step` | int32 | decode step; **0 for all of prefill** |
| `layer` | int16 | MoE layer index |
| `seq_id` | int16 | sequence within the batch, `0..batch-1` |
| `pos` | int32 | token position in the sequence |
| `slot` | int8 | rank within the top-k, `0..k-1` |
| `expert` | int16 | **the expert that was selected** |
| `weight` | float32 | router probability for that expert |

A single token at a single layer occupies **k rows**, so

```
len(df) == batch * (prompt_len + gen_len) * len(moe_layers) * top_k
```

Format is Parquet + zstd, columnar: OLMoE's 6.96M records occupy 14 MB. Readable
from pandas, pyarrow, polars, DuckDB, or Spark.

## Reading it

```python
import pandas as pd
df = pd.read_parquet("data/routing/OLMoE-1B-7B/b1_p512_g128.parquet")
```

Filters and column selection push down to the file, so you never pay to read
what you skip:

```python
pd.read_parquet(path,
                columns=["layer", "expert"],
                filters=[("phase", "==", 1), ("layer", "==", 0)])
```

Recovering the expert set per token — the form `analytical/trace_gen.py` wants:

```python
sets = (df[df.phase == 1]
        .groupby(["step", "seq_id", "layer"])["expert"]
        .apply(list).reset_index())
# step seq_id layer  expert
#    0      0     0  [19, 10, 57, 9, 7, 61, 33, 60]
```

## What the parquet cannot say

A routing log records *which expert*, never *how large it is*. That comes from
`mapping/geometry/<model>.json`: `num_experts`, `top_k`, `hidden_size`,
`expert_ff`, `moe_layers`, `num_shared_experts`, `vocab_size`. The model
multiplies the two — expert id → address range → bytes → transfer time — which
is why a routing log alone is not enough to run anything.

`routing/index.json` supplies the rest: for each completed run, its `batch`,
`prompt_len`, `gen_len`, `rows`, and `file`.

## Traps

**Filter on `phase` first.** Prefill carries `step = 0` for every token —
position lives in `pos`, not `step`. Filtering on `step == 0` silently returns
the entire prompt plus the first decode token.

**`slot` is rank, not order of use.** Slot 0 is the highest-weighted expert for
most models, but DeepSeek's router calls `topk(sorted=False)`, so its slot order
is arbitrary. Sort by `weight` if you need the true top-1.

**Shared experts are absent by design.** DeepSeek-V2-Lite has 2 shared experts
that every token uses at every layer. They are not routing *decisions*, so no
router emits them and none were recorded. They appear only in the metadata
(`num_shared_experts: 2`, 33.0 MiB per layer). The address map adds them as
permanently resident; without that DeepSeek's per-layer footprint is
understated.

**Timing validity varies — and none of it is the number you want.**
`layer_timing.valid` is false for Mixtral and Phi-3.5-MoE: both were fully
offloaded, so their ~50 s per decode step is NVMe streaming, not compute. Only
GPU-resident layers give real figures (OLMoE 0.949, DeepSeek 1.03, Qwen dense
0.401 ms/layer), and even those are optimistic — `transformers` runs the MoE
layer as a Python loop over hit experts, roughly 5x slower than the
bandwidth-bound floor.

More importantly, an RTX 4070 is not the target hardware for any of this. Layer
time for a datacenter GPU should be **derived** (`bytes read per layer / HBM
bandwidth`) and swept as a parameter; the measured values serve only as a
sanity check that the analytic form is right. Note the direction of the effect:
a *faster* GPU shortens the compute window and therefore makes prefetching
**harder**, not easier.

**Run configurations are not perfectly uniform.** DeepSeek's batch-64 run used a
256-token prompt rather than 512, because its KV cache is stored decompressed
(270 KB/token) and 576 tokens x 64 sequences would not fit in 11.6 GiB. The two
coarse models used 256-token prompts and 32 generated tokens throughout. Prompt
length, generation length and prefill chunk are recorded per run in
`routing/index.json` — read them there rather than assuming.

## Verifying

```bash
make routing                                   # all models
python data/check_routing.py --tag OLMoE-1B-7B  # just one
```

Checks, per file: expert ids in range; exactly the MoE layers present; exactly
`k` rows per (sequence, layer, position) with no repeated expert; every prefill
position present exactly once with uniform counts; decode `pos == prompt_len +
step`; all batch sequences present; weights finite and non-negative. All 20
files (5 models x 4 batch sizes) currently pass.

## Provenance

Prompts are fixed-length token windows cut from **wikitext-2-raw-v1**, so no
padding and therefore no garbage routing from pad positions. Decoding uses
sampling (`temperature=0.8, top_p=0.95`, seed 1234) rather than greedy — greedy
decoding collapses into repetition, which would manufacture temporal locality in
the expert stream.

Everything ran in **bf16**, so routing is numerically exact; there is no
quantization caveat. Models larger than VRAM were offloaded to CPU and NVMe,
which costs speed but not accuracy.

The corpus cache is keyed per tokenizer. An earlier shared cache fed one model's
token ids to another — ids stay in range whenever the second vocabulary is
larger, so nothing crashed, but the text was noise under the other tokenizer and
*inflated apparent routing skew*. Those runs were discarded and re-collected.
The collection code asserts every id lies inside the model's embedding matrix.

The prompt corpus and the collection scripts are not part of this repository —
the logs are shipped instead, because re-collecting them needs the checkpoints
(87 GiB for Mixtral alone) and a GPU, while consuming them needs neither.
