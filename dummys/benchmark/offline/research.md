# Research Report: `swiftLLM/examples`

## 1. Scope

This report analyzes the `swiftLLM/examples` folder in depth, with the main focus on [`examples/offline.py`](../examples/offline.py). It also uses the underlying implementation files in `swiftLLM/swiftllm/` to explain what the example is actually doing at runtime.

The folder contains only two example scripts:

- [`examples/offline.py`](../examples/offline.py): direct, data-plane-only inference through `swiftllm.LlamaModel`
- [`examples/online.py`](../examples/online.py): engine-based inference through `swiftllm.Engine`

Per the project README, the intended distinction is:

- `offline.py` is the starting point if you want to use **only the data plane**.
- `online.py` is the starting point if you want to use **both the control plane and the data plane**.

That mapping is confirmed by [`README.md`](../README.md), [`swiftllm/__init__.py`](../swiftllm/__init__.py), and the implementation of the model and engine.

---

## 2. What `offline.py` does at a high level

[`examples/offline.py`](../examples/offline.py) is a minimal script that:

1. Parses `--model-path`
2. Builds a `swiftllm.EngineConfig`
3. Instantiates `swiftllm.LlamaModel` directly
4. Loads weights from a local model directory
5. Profiles how many GPU KV-cache blocks can fit in memory
6. Allocates GPU KV cache and CPU swap tensors
7. Tokenizes a fixed list of prompts
8. Runs one **prompt/prefill** forward pass
9. Runs 20 **decode** forward passes, one token per sequence per step
10. Decodes and prints generated text

So this example is not a full server, not a scheduler, and not a request manager. It is a direct demonstration of a single model worker / executor path.

The script description says exactly that: it demonstrates how to use the model executor directly "without using the engine".

---

## 3. CLI contract and immediate assumptions

`offline.py` accepts exactly one CLI argument:

- `--model-path` (required)

This path must point to a **local model directory**. The example and README both assume you downloaded model assets in advance. SwiftLLM does **not** auto-download from Hugging Face in this flow.

### Files implicitly expected under `model_path`

Based on the code path in [`swiftllm/model_config.py`](../swiftllm/model_config.py) and [`swiftllm/worker/weight.py`](../swiftllm/worker/weight.py), the directory is expected to contain at least:

- `config.json`
- tokenizer files usable by `transformers.AutoTokenizer.from_pretrained(model_path)`
- either:
  - `.safetensors` weights, optionally with `model.safetensors.index.json`, or
  - PyTorch `.bin` weights, optionally with `pytorch_model.bin.index.json`

### Model-type assumptions

`LlamaModelConfig` asserts that the model config has:

- `model_type == "llama"`
- `hidden_act == "silu"`

So this example is not a generic Hugging Face model loader. It is specifically built around the LLaMA family assumptions encoded in SwiftLLM.

---

## 4. The exact config used by `offline.py`

`offline.py` constructs this `EngineConfig`:

- `model_path = model_path`
- `use_dummy = False`
- `block_size = 16`
- `gpu_mem_utilization = 0.99`
- `num_cpu_blocks = 0`
- `max_seqs_in_block_table = 128`
- `max_blocks_per_seq = 2048`
- `max_batch_size = 16`
- `max_tokens_in_batch = 2048 * 16` (32768)

### Important specifics

#### `use_dummy = False`
Real weights are loaded from disk. If set to `True`, SwiftLLM would generate random CUDA tensors instead of loading actual checkpoint weights.

#### `block_size = 16`
KV cache is paged in units of 16 tokens per block.

#### `gpu_mem_utilization = 0.99`
The profiler tries to reserve almost all GPU memory for runtime + KV cache. This is aggressive.

#### `num_cpu_blocks = 0`
Offline mode disables practical CPU swap capacity. The code still creates CPU swap tensors and a CPU block manager, but both are sized to zero blocks, so swap-in/swap-out is effectively unavailable.

#### `max_seqs_in_block_table = 128`
At most 128 sequence IDs can be represented in the block table.

#### `max_blocks_per_seq = 2048`
A single sequence can reserve at most 2048 blocks. With block size 16, that implies a theoretical tracked sequence length ceiling of about 32768 tokens in the block manager.

#### `max_batch_size = 16`, `max_tokens_in_batch = 32768`
The code comment says these are "not used in the offline example", but that is not strictly true.

They are **used inside `LlamaModel.profile_num_blocks()`** to synthesize a maximum-size prefill batch and estimate GPU KV capacity. They are not used for request scheduling like they are in the engine path, but they absolutely affect memory profiling and therefore the final number of cache blocks.

This is one of the most important non-obvious specifics in the script.

### How these values differ from the generic CLI defaults

The generic `EngineConfig.add_cli_args()` defaults in [`swiftllm/engine_config.py`](../swiftllm/engine_config.py) are much larger / different in some places:

- `gpu_mem_utilization` default there is `0.97`, not `0.99`
- `num_cpu_blocks` default there is `2048`, not `0`
- `max_seqs_in_block_table` default there is `4096`, not `128`
- `max_blocks_per_seq` default there is `32768`, not `2048`
- `max_batch_size` default there is `512`, not `16`
- `max_tokens_in_batch` default there is `32768`, same total as offline’s chosen value

So `offline.py` is a deliberately narrower, simpler configuration than the engine’s general-purpose CLI settings.

---

## 5. Initialization contract of `LlamaModel`

The intended initialization order is documented directly in [`swiftllm/worker/model.py`](../swiftllm/worker/model.py):

1. call `__init__()`
2. call `load_weights()`
3. call `profile_num_blocks()`
4. call `init_kvcache_and_swap()`

`offline.py` follows this exactly:

```python
model = swiftllm.LlamaModel(engine_config)
model.load_weights()
num_blocks = model.profile_num_blocks()
model.init_kvcache_and_swap(num_blocks)
```

This sequence is mandatory because:

- `__init__()` loads static model metadata from `config.json`
- `load_weights()` builds the actual layers and weight tensors
- `profile_num_blocks()` needs a usable model to measure runtime memory usage
- `init_kvcache_and_swap()` cannot size KV cache until the number of blocks is known

### What `__init__()` actually sets up

Inside [`swiftllm/worker/model.py`](../swiftllm/worker/model.py):

- stores the `EngineConfig`
- loads `LlamaModelConfig` from `model_path/config.json`
- initializes placeholders for:
  - weights
  - rotary caches (`_cos_cached`, `_sin_cached`)
  - pre-layer / transformer layers / post-layer
  - KV cache tensors
  - swap tensors
  - GPU and CPU block managers

At this point, the model object exists, but it is not ready to run inference yet.

---

## 6. Weight loading behavior

Weight loading is handled by `load_weights(...)` in [`swiftllm/worker/weight.py`](../swiftllm/worker/weight.py).

### Supported formats

SwiftLLM supports both:

- **Safetensors** checkpoints
- **PyTorch `.bin`** checkpoints

and both can be:

- single-file
- sharded with an index JSON

### Loading logic

The loader does this:

1. Detect model version
   - if `rope_scaling` in `config.json` is a dictionary, it treats it as `llama3.2`
   - otherwise it treats it as `llama`
2. Choose weight source
   - `.safetensors` if any such files exist
   - otherwise PyTorch `.bin`
3. Build a `LlamaWeight` object
4. Register all expected tensor names and shapes
5. Load them directly onto CUDA
6. Post-process some weights

### Non-obvious specifics

#### All loaded weights are expected on GPU
The loader asserts `weight_value.device.type == "cuda"`. This code path is GPU-only.

#### Layer weights are shape-checked
Each registered tensor is validated against an expected shape. That makes mismatch failures explicit.

#### `up_proj` and `gate_proj` are concatenated after loading
`LlamaTransformerLayerWeight._post_process_after_load()` concatenates them into `up_gate_proj`, which the FFN path later uses more efficiently.

#### Llama 3.2 has a special `lm_head` mapping
For `llama3.2`, `lm_head` is mapped to `model.embed_tokens.weight` instead of `lm_head.weight`.

#### PyTorch `.bin` files are memoized after first load
The code keeps opened `.bin` files in a dictionary because deserialization is expensive, and uses `mmap=True`.

#### Safetensors files are reopened per tensor access
The comment explicitly says opening safetensors is cheap, so they reopen the file each time.

---

## 7. Rotary embedding cache behavior

During `model.load_weights()`, the model also initializes rotary embedding caches by calling `_init_to_get_rotary()`.

This method in [`swiftllm/worker/model.py`](../swiftllm/worker/model.py) precomputes:

- `self._cos_cached`
- `self._sin_cached`

### Specifics

- The cache is created on CUDA.
- It supports both:
  - scalar `rope_scaling`
  - dictionary-style `rope_scaling` used by Llama 3.2
- It extends the cached sequence length to `max_seq_len + 128`

This means inference later can index precomputed cosine/sine values by token position instead of recomputing them on every forward pass.

---

## 8. GPU block profiling: what `profile_num_blocks()` really measures

After loading weights, `offline.py` calls:

```python
num_blocks = model.profile_num_blocks()
```

This is one of the most important steps in the entire example.

### What it does

`profile_num_blocks()` in [`swiftllm/worker/model.py`](../swiftllm/worker/model.py):

1. Clears CUDA cache and peak stats
2. Synthesizes a fake **maximum-size prefill batch** using:
   - `engine_config.max_tokens_in_batch`
   - `engine_config.max_batch_size`
3. Builds fake token IDs filled with zeros
4. Runs `self.forward(..., ignore_kvcache=True)`
5. Synchronizes CUDA
6. Measures memory usage via `torch.cuda.mem_get_info()`
7. Computes how many KV blocks can fit in remaining usable memory

### Why the forged batch matters

This step assumes that the worst-case runtime memory footprint can be estimated by running a large prefill batch. The reported number of KV blocks is therefore **configuration-dependent**.

In `offline.py`, because:

- `max_batch_size = 16`
- `max_tokens_in_batch = 32768`

it profiles using a synthetic prefill workload of 16 sequences totaling 32768 prompt tokens.

### Exact block-sizing formula

The block byte size is:

```text
block_size * kvslot_size
```

where `kvslot_size` from [`swiftllm/model_config.py`](../swiftllm/model_config.py) is:

```text
2 * num_layers * num_kv_heads * head_dim * dtype.itemsize
```

The factor of 2 accounts for both K and V.

Then:

```text
num_gpu_blocks = floor((usable_memory - peak_memory) / block_size_bytes)
```

with:

```text
usable_memory = total_memory * gpu_mem_utilization
```

### Important failure condition

If the profiled runtime peak already exceeds `usable_memory`, SwiftLLM raises a runtime error.

### Important caveat

This is an **estimate**, not an exhaustive proof of safety for every later workload. It is designed as a practical memory sizing heuristic.

---

## 9. KV cache and swap allocation

After profiling, `offline.py` calls:

```python
model.init_kvcache_and_swap(num_blocks)
```

### GPU KV cache layout

`init_kvcache_and_swap()` allocates `k_cache` and `v_cache` with shape:

```text
(num_blocks,
 num_layers,
 num_kv_heads,
 block_size,
 head_dim)
```

Important detail: these tensors are created with `torch.zeros`, not `torch.empty`.

The comment explains why: uninitialized memory may contain NaNs, and NaNs could propagate into model outputs.

### CPU swap layout

It also allocates `k_swap` and `v_swap` on CPU with shape:

```text
(num_cpu_blocks,
 num_layers,
 num_kv_heads,
 block_size,
 head_dim)
```

In `offline.py`, `num_cpu_blocks = 0`, so these are effectively zero-capacity swap tensors.

### Block managers

Two `BlockManager` instances are created:

- GPU block manager
- CPU block manager

Each block manager tracks:

- per-sequence allocated block counts
- block table `(seq_id, block_index) -> block_id`
- free block bitmap

Per [`swiftllm/worker/block_manager.py`](../swiftllm/worker/block_manager.py), these tracking structures are maintained on CUDA, even for the CPU-side block manager metadata.

This means allocation bookkeeping is GPU-side for speed, even though CPU swap storage itself is on host memory.

---

## 10. Tokenization path in `offline.py`

`offline.py` tokenizes with:

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer(prompts)['input_ids']
```

### Specifics

- It relies on Hugging Face tokenizer assets living under the same `model_path`.
- It directly asks for `input_ids`; no padding / truncation options are set here.
- The result is a Python `list[list[int]]`, not a PyTorch tensor yet.

That list-of-lists is exactly what `LlamaModel.forward()` expects.

---

## 11. The first forward pass: prompt / prefill phase

The prompt phase is:

```python
prompt_phase_outputs = model.forward(
    input_ids,
    list(range(0, len(prompts))),
    []
)
```

### Meaning of the arguments

#### `input_ids`
A batch of full prompt token sequences.

#### `seq_ids`
The sequence IDs are `[0, 1, 2, 3]` for the four hardcoded prompts.

These are not token positions. They are identifiers used by the block manager to track per-sequence KV allocation.

#### `decoding_seq_lens = []`
This indicates there are **no decode-stage sequences** in this batch.

So this whole batch is treated as **prefill**.

### What `forward()` returns

It returns **one token ID per sequence**.

This is a critical detail: even during prompt phase, the model does not return full logits for every token. The post-layer slices out the final hidden state for each sequence and runs only those through final norm + LM head + argmax.

So the prompt phase output is the **first generated token after each full prompt**.

---

## 12. The decode loop in `offline.py`

After the prompt phase, the script runs:

```python
for _ in range(20):
    for i, _ in enumerate(prompts):
        seq_lens[i] += 1
    last_round_outputs = model.forward(
        [[x] for x in last_round_outputs],
        list(range(0, len(prompts))),
        seq_lens
    )
    outputs.append(last_round_outputs)
```

### What this means

For 20 iterations:

- each sequence length is incremented by 1
- the input batch contains exactly one token per sequence: the token generated in the previous step
- `decoding_seq_lens` now supplies the **current total sequence lengths**

This is pure autoregressive decoding.

### Total generated tokens

The script stores:

- 1 token from prompt phase
- 20 tokens from decode phase

So each prompt gets **21 generated tokens total**.

### No stopping criteria

There is:

- no EOS check
- no max-new-token arg
- no per-sequence early stop
- no sampling parameters

The script simply forces 20 decode rounds regardless of content.

---

## 13. `LlamaModel.forward()` contract and batching semantics

The core behavior of the example depends on `LlamaModel.forward()` in [`swiftllm/worker/model.py`](../swiftllm/worker/model.py).

Its signature is conceptually:

```python
forward(input_ids_list, seq_ids_list, decoding_seq_lens_list, ignore_kvcache=False)
```

### Most important contract

The batch may contain a mixture of:

- prefill sequences
- decode sequences

but the code assumes the ordering is:

- **prefill sequences first**
- **decode sequences last**

This is stated implicitly by the implementation:

```python
num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
```

and by the comments in the file.

So `forward()` is not a general unordered mixed batch interface. It expects a specific layout.

### What the method builds

It:

1. Flattens the nested token lists into one 1D token stream
2. Builds sequence length tensors
3. Computes prefill start offsets
4. Builds per-token position indices
5. Allocates blocks if KV cache is active
6. Chooses a decode paged-attention `seq_block_size`
7. Packs everything into `LlamaInferState`
8. Calls `_forward()`

### Position indices behavior

For prefill tokens, positions are expanded like:

```text
[0, 1, 2, ..., prefill_len - 1]
```

for each sequence.

For decode tokens, the position is:

```text
decode_len - 1
```

for each sequence, because each decode sequence contributes exactly one new token per step in this path.

These positions are used for RoPE lookup.

---

## 14. KV block allocation during forward

Unless `ignore_kvcache=True`, `forward()` calls:

```python
self.gpu_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
```

This ensures each sequence has enough page blocks to hold its current token length.

### Important consequences

- During the prompt phase, blocks are allocated to fit the whole prompt length.
- During decode, more blocks are allocated only when a sequence crosses a block boundary.
- Allocation is per sequence ID, so consistent `seq_ids` across steps is essential.

If you changed sequence IDs between steps, the model would treat them as unrelated sequences and KV tracking would break semantically.

---

## 15. Decode `seq_block_size` heuristic

`forward()` chooses a `seq_block_size` for paged attention using a heuristic in [`swiftllm/worker/model.py`](../swiftllm/worker/model.py).

It starts at 2048 and may halve it while trying to keep enough useful thread blocks to utilize the GPU, without making reduction overhead explode.

The comments explain the logic in detail:

- paged attention phase 1 launches blocks over sequence chunks
- more chunks increase parallelism
- too many chunks increase phase-2 reduction cost

The code aims for roughly at least 1024 useful blocks as a heuristic.

This is one of the more performance-engineered parts of the implementation.

---

## 16. What `_forward()` actually does layer by layer

`_forward()` in [`swiftllm/worker/model.py`](../swiftllm/worker/model.py) performs the real model computation.

### Step 1: embedding lookup

`LlamaPreLayer.forward()` in [`swiftllm/worker/layers/pre_layer.py`](../swiftllm/worker/layers/pre_layer.py):

- uses `torch.embedding(self.weights.wte, input_ids, padding_idx=-1)`
- does not use `nn.Embedding`
- avoids duplicating the already-loaded embedding weights

### Step 2: transformer layers

For every layer, `LlamaTransformerLayer.forward()` in [`swiftllm/worker/layers/transformer_layer.py`](../swiftllm/worker/layers/transformer_layer.py) does:

1. fused residual add + RMSNorm
2. Q / K / V projections
3. reshape into attention-head views
4. in-place rotary embedding
5. optional KV-cache store
6. prefill attention for prompt tokens
7. paged attention for decode tokens
8. output projection
9. fused residual add + RMSNorm again
10. FFN using fused `up_gate_proj`, SiLU-and-multiply, then `down_proj`

### Attention split: prefill vs decode

This is a key design choice.

#### Prefill path
If there are prompt-phase tokens, the code uses:

- `vllm_flash_attn.flash_attn_varlen_func(...)`

This means SwiftLLM is intentionally delegating prompt-phase flash attention to the vLLM flash attention implementation because the author comments that its performance is better there.

#### Decode path
If there are decode sequences, the code uses SwiftLLM’s own:

- `paged_attention(...)`

This runs on a separate CUDA stream named `decoding_piggyback_stream`.

### Why the extra CUDA stream exists

The comments describe a concurrency strategy:

- default CUDA stream handles KV-cache storage and prefill-side work
- a second stream handles decode paged attention
- decode waits for KV storage to complete via CUDA events

So the layer is explicitly designed to overlap work where possible.

---

## 17. Post-layer behavior: why only one token per sequence comes back

`LlamaPostLayer.forward()` in [`swiftllm/worker/layers/post_layer.py`](../swiftllm/worker/layers/post_layer.py) does not compute sampled outputs for every token in the flattened batch.

Instead, it:

1. computes the index of the **last token of each sequence**
2. gathers only those hidden states
3. applies final RMSNorm
4. applies `lm_head`
5. returns `torch.argmax(logits, dim=1)`

### Consequences

- Output shape is `[batch_size]`, one token per sequence.
- This is greedy decoding only.
- There is no temperature, top-k, top-p, repetition penalty, or sampling logic in this example path.

This explains why `offline.py` can append each `forward()` result as a single generation step.

---

## 18. How output reconstruction works in `offline.py`

The script keeps a list named `outputs` where each element is a batch output:

- `outputs[0]` = prompt-phase first generated token for every prompt
- `outputs[1]`..`outputs[20]` = decode-step outputs

Then for each prompt `i`, it reconstructs that prompt’s generated sequence via:

```python
output_tokens = [x[i] for x in outputs]
output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
```

### Important implications

- It decodes only the generated tokens, **not** the original prompt.
- Final printed text is:

```text
{prompt}|{generated_text}
```

So the script visually pairs the prompt with the generated continuation, but the right-hand decoded string does not include the prompt tokens themselves.

---

## 19. What `online.py` does differently

[`examples/online.py`](../examples/online.py) uses a much higher-level interface:

- constructs `swiftllm.Engine`
- calls `await engine.initialize()`
- starts background event loops
- submits `RawRequest(prompt, output_len)` objects
- either streams tokens or waits for full completion

### What the engine adds on top

From [`swiftllm/server/engine.py`](../swiftllm/server/engine.py), the engine provides:

- request objects and request IDs
- tokenization batching in a tokenization engine
- a scheduler
- async event loops
- streaming mode
- non-streaming mode
- sequence resource cleanup on completion
- swap-in / swap-out orchestration

### Engine initialization still uses the same model core

`Engine.initialize()` internally does:

1. `self.model = LlamaModel(...)`
2. `self.model.load_weights()`
3. `self.model.profile_num_blocks()`
4. `self.model.init_kvcache_and_swap(...)`

So the online path and offline path share the same low-level model implementation. The difference is that the engine wraps it with scheduling and async request management.

### Config differences in the example

Compared with `offline.py`, `online.py` changes several values:

- `num_cpu_blocks = 1024` instead of `0`
- `max_blocks_per_seq = 3072` instead of `2048`
- `max_batch_size = 4` instead of `16`
- `max_tokens_in_batch = 1024` instead of `32768`
- adds optional `--streaming`

This makes sense because the online example is demonstrating engine behavior rather than maximum-size direct forward profiling.

---

## 20. Important caveats and edge cases in `offline.py`

### 20.1 CUDA is mandatory
Many tensors are hard-coded to `device="cuda"` throughout model, weight, and block-manager code. There is no CPU fallback in this path.

### 20.2 The tokenizer must be available locally
`AutoTokenizer.from_pretrained(model_path)` assumes tokenizer files exist at the same local path.

### 20.3 It is LLaMA-specific
The config loader asserts a LLaMA-style config. This is not a universal Transformer runner.

### 20.4 The script is greedy only
No sampling strategy is exposed or implemented in the example output path.

### 20.5 The script never stops early
There is no EOS handling. It will always run exactly 20 decode iterations after the prompt phase.

### 20.6 The prompt list is hardcoded
The example is not a reusable CLI inference tool. It is a code example with fixed prompts.

### 20.7 Sequence IDs matter
The same sequence IDs are reused across prompt and decode steps. This is required for KV-cache continuity.

### 20.8 `num_cpu_blocks = 0` disables meaningful swap capacity
The model still creates swap-related objects, but offline example behavior is effectively GPU-only residency.

### 20.9 The comment about unused scheduling fields is misleading
`max_batch_size` and `max_tokens_in_batch` are indeed unused for request scheduling in this script, but they are still used during KV-block profiling.

### 20.10 Prefill and decode ordering is a real contract
If someone reused `forward()` incorrectly with mixed ordering, the inferred number of prefill vs decode sequences would be wrong.

### 20.11 Model creation time includes multiple phases
The printed `model_creation_time` is not just constructor time. It includes:

- model object creation
- weight loading
- GPU block profiling
- KV-cache and swap allocation

So it is more like "model initialization time".

---

## 21. Practical interpretation of `offline.py`

Conceptually, `offline.py` is demonstrating the smallest complete manual usage of SwiftLLM’s inference worker:

- load a LLaMA-family checkpoint
- size and allocate paged KV cache
- run one prompt prefill step
- repeatedly run one-token decode steps
- keep per-sequence state via sequence IDs and KV blocks

It is therefore best understood as:

- a low-level integration example
- a debugging / experimentation starting point
- a data-plane demonstration

It is **not** meant to be a polished inference application.

---

## 22. Summary of the most important findings

1. `offline.py` uses `swiftllm.LlamaModel` directly and bypasses the engine entirely.
2. The required initialization sequence is explicit and must be followed exactly.
3. The model loader is LLaMA-specific and GPU-only.
4. The example supports local `.safetensors` and `.bin` checkpoints, including sharded ones.
5. KV-cache capacity is derived by profiling a forged maximum-size prefill batch.
6. `max_batch_size` and `max_tokens_in_batch` materially affect that profiling result, despite the script comment implying otherwise.
7. The forward API expects prefill sequences first and decode sequences last in mixed batches.
8. Prefill attention uses vLLM flash attention, while decode uses SwiftLLM paged attention.
9. The post layer performs greedy next-token selection only.
10. The example generates exactly 21 tokens per prompt in total: 1 from prompt phase + 20 decode steps.
11. `online.py` wraps the same low-level model core with async request handling, tokenization, scheduling, and optional streaming.

---

## 23. Bottom line

If your goal is to understand SwiftLLM’s core inference path, [`examples/offline.py`](../examples/offline.py) is the clearest entry point in the repository because it exposes the low-level lifecycle directly:

- config
- model construction
- weight loading
- memory profiling
- KV-cache allocation
- prompt prefill
- incremental decode

If your goal is to understand how SwiftLLM serves multiple requests and manages them asynchronously, [`examples/online.py`](../examples/online.py) is the next step, because it layers an engine, scheduler, tokenization, and streaming interface on top of the same underlying `LlamaModel`.
