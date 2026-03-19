"""
================================================================================
PagedAttention 注释版本
================================================================================
 
核心思想:
    传统 KV Cache 要求为每条序列预分配一段连续的内存（最大序列长度 × head_dim × 2），
    浪费严重。PagedAttention 借鉴操作系统"虚拟内存分页"的思路：
      - 将 KV Cache 切成固定大小的"物理块"（block），每块存 block_size 个 token 的 K/V
      - 用 block_table 建立"序列逻辑块号 → 物理块号"的映射
      - 序列不需要连续内存，按需申请块，大幅提高显存利用率
 
贯穿全文的具体例子（请牢记这些数字）：
    - num_decoding_seqs = 2       # 2 条解码中的序列
    - num_q_heads = 4             # 每个 token 4 个 Q head
    - num_kv_heads = 2            # GQA: 2 个 KV head，Q/KV head 比 = 2
    - head_dim = 64               # 每个 head 的维度
    - block_size = 4              # 每个物理块存 4 个 token 的 KV
    - seq_block_size = 8          # Phase1 每个 Triton kernel 负责 8 个 token
    - 序列长度: seq_lens = [10, 6]
    - num_seq_blocks = ceil(10/8) = 2   # 按最长序列决定
    - num_layers = 32
    - max_blocks_per_seq = ceil(max_seq_len/block_size) = ceil(10/4) = 3
 
    block_table (shape [num_seqs, max_blocks_per_seq]):
        seq0: [逻辑块0→物理块5, 逻辑块1→物理块12, 逻辑块2→物理块3]
              即 token[0:4]在物理块5, token[4:8]在物理块12, token[8:10]在物理块3
        seq1: [逻辑块0→物理块7, 逻辑块1→物理块2, ...]
              即 token[0:4]在物理块7, token[4:6]在物理块2
 
    k_cache / v_cache shape:
        [num_total_blocks, num_layers, num_kv_heads, block_size, head_dim]
        例: [20, 32, 2, 4, 64]  (假设总共分配了 20 个物理块)
 
整体执行流程:
    Phase 1 (并行): grid = [2, 4, 2]，即 2×4×2 = 16 个 Triton kernel 同时运行
        每个 kernel 负责: 某条序列 × 某个 Q head × 某段 seq_block
        计算该 seq_block 内的局部 softmax 加权 V，输出到 mid_o / mid_o_logexpsum
 
    Phase 2 (规约): grid = [2, 4]，即 8 个 Triton kernel
        每个 kernel 负责: 某条序列 × 某个 Q head
        把 Phase 1 产出的多个局部结果合并（online softmax），输出最终 o
"""

import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState

# ==============================================================================
# Phase 1: 局部注意力计算
# ==============================================================================
 
@triton.jit
def _fwd_paged_attention_phase1(
    # --------------------------------------------------------------------------
    # 输出张量
    # --------------------------------------------------------------------------
    mid_o,          # shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
                    # 存储每个 seq_block 的局部 attention 输出（已除以局部 sum_exp，但还未全局归一化）
                    # 例: [2, 4, 2, 64]，共 2×4×2=16 个局部结果，每个是 64 维向量
    mid_o_logexpsum,# shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
                    # 存储每个 seq_block 的 log-sum-exp（用 log2 表示，下面解释为何用 log2）
                    # 例: [2, 4, 2]，每个值是一个标量
 
    # --------------------------------------------------------------------------
    # 输入张量
    # --------------------------------------------------------------------------
    q,              # shape: [num_decoding_seqs, num_q_heads, head_dim]
                    # 当前 step 的 Query，解码阶段每条序列只有 1 个新 token
                    # 例: [2, 4, 64]
 
    k_cache,        # shape: [num_total_blocks, num_layers, num_kv_heads, block_size, head_dim]
                    # 所有物理块的 K Cache，非连续存储
                    # 例: [20, 32, 2, 4, 64]
 
    v_cache,        # shape: [num_total_blocks, num_layers, num_kv_heads, block_size, head_dim]
                    # 所有物理块的 V Cache
                    # 例: [20, 32, 2, 4, 64]
 
    block_table,    # shape: [num_seqs_in_engine, max_blocks_per_seq]
                    # 逻辑块 → 物理块的映射表
                    # 例: [[5,12,3], [7,2,?], ...]（? 表示未用到的槽位）
                    # 注意：这里是全引擎所有序列的 block_table，用 seq_id 索引
 
    softmax_scale,  # 标量，= (1/sqrt(head_dim)) * log2(e)
                    # 例: (1/sqrt(64)) * 1.4427 = 0.125 * 1.4427 ≈ 0.1803
                    # 之所以预乘 log2(e)：因为后面用 exp2 代替 exp（见下方说明）
 
    decoding_seq_lens,  # shape: [num_decoding_seqs]，每条解码序列的当前长度
                        # 例: [10, 6]
 
    seq_ids,        # shape: [num_decoding_seqs]，每条解码序列在全引擎中的 seq_id
                    # 用于查 block_table（因为 block_table 是按全引擎 seq_id 索引的）
                    # 例: [3, 7]（表示 batch 中第0条是引擎全局 seq3，第1条是 seq7）
 
    num_seq_blocks, # 标量，= ceil(max_seq_len / seq_block_size)，例: ceil(10/8) = 2
    cur_layer,      # 当前 Transformer 层号，例: 15（从 k_cache 中取第15层的 K）
 
    # --------------------------------------------------------------------------
    # 编译期常量（tl.constexpr：Triton 在 JIT 编译时内联，不占寄存器）
    # --------------------------------------------------------------------------
    num_layers:     tl.constexpr,   # 例: 32
    num_q_heads:    tl.constexpr,   # 例: 4
    num_kv_heads:   tl.constexpr,   # 例: 2
    num_my_heads:   tl.constexpr,   # = num_q_heads // num_kv_heads，例: 2（GQA 分组大小）
    block_size:     tl.constexpr,   # 例: 4
    head_dim:       tl.constexpr,   # 例: 64
    seq_block_size: tl.constexpr,   # 例: 8
    max_blocks_per_seq: tl.constexpr,  # 例: 3
):
    # --------------------------------------------------------------------------
    # 1. 确定"我是谁"：从 grid 坐标读取本 kernel 的工作范围
    # --------------------------------------------------------------------------
    # grid = (num_decoding_seqs, num_q_heads, num_seq_blocks) = (2, 4, 2)
    # 共 16 个 kernel 并行，每个 kernel 处理一个 (batch, q_head, seq_block) 三元组
 
    my_batch_id = tl.program_id(0).to(tl.int64)
    # 本 kernel 处理第几条序列（在 decoding batch 中的下标）
    # 例：my_batch_id = 0 → 处理 seq_lens=10 的那条序列
 
    my_q_head_id = tl.program_id(1).to(tl.int64)
    # 本 kernel 处理哪个 Q head，例: my_q_head_id = 2
 
    my_seq_block_id = tl.program_id(2)
    # 本 kernel 处理该序列的第几个 seq_block，例: my_seq_block_id = 1（第二段，token[8:10]）
 
    my_kv_head_id = my_q_head_id // num_my_heads
    # GQA：多个 Q head 共享同一个 KV head
    # 例: my_q_head_id=2, num_my_heads=2 → my_kv_head_id=1
    # 即 Q head 0,1 → KV head 0；Q head 2,3 → KV head 1
 
    # --------------------------------------------------------------------------
    # 2. 加载本序列的元信息
    # --------------------------------------------------------------------------
    my_seq_id = tl.load(seq_ids + my_batch_id)
    # 从 seq_ids 取出本序列在引擎全局的 seq_id，用于查 block_table
    # 例: my_batch_id=0, seq_ids=[3,7] → my_seq_id=3
 
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    # 本序列实际长度，例: my_seq_len = 10
 
    my_start_token_idx = my_seq_block_id * seq_block_size
    # 本 seq_block 负责的起始 token 下标
    # 例: my_seq_block_id=1, seq_block_size=8 → my_start_token_idx=8（处理 token[8:10]）
 
    # --------------------------------------------------------------------------
    # 3. 越界检查：如果本 seq_block 起始位置已超出序列长度，直接退出
    # --------------------------------------------------------------------------
    if my_start_token_idx >= my_seq_len:
        return
    # 例：序列长度 6，seq_block_id=1 时 start=8 >= 6，该 kernel 直接退出
    # （这保证了短序列不会读取不属于自己的 KV 数据）
 
    # --------------------------------------------------------------------------
    # 4. 加载 Query 向量
    # --------------------------------------------------------------------------
    offs_q = (my_batch_id * num_q_heads * head_dim   # 跳到第 my_batch_id 条序列
              + my_q_head_id * head_dim               # 跳到第 my_q_head_id 个 head
              + tl.arange(0, head_dim))               # 取整个 head_dim 向量
    # 例: my_batch_id=0, my_q_head_id=2, head_dim=64
    #     offs_q = 0*4*64 + 2*64 + [0..63] = [128, 129, ..., 191]
    my_q = tl.load(q + offs_q)  # shape: [head_dim]，例 shape [64]
 
    # --------------------------------------------------------------------------
    # 5. 计算 K/V Cache 的基础指针
    # --------------------------------------------------------------------------
    # k_cache layout: [num_total_blocks, num_layers, num_kv_heads, block_size, head_dim]
    # 对于给定的 (cur_layer, my_kv_head_id)，在块内的偏移是固定的：
    #   块内偏移 = (cur_layer * num_kv_heads + my_kv_head_id) * block_size * head_dim
    # 物理块 block_index 带来的偏移需要在循环中动态加入
 
    start_block_idx = my_seq_block_id * (seq_block_size // block_size)
    # 本 seq_block 对应的起始逻辑块号
    # 例: my_seq_block_id=1, seq_block_size=8, block_size=4
    #     start_block_idx = 1 * (8//4) = 2，即从逻辑块2开始
 
    k_ptrs = (k_cache
              + (cur_layer * num_kv_heads + my_kv_head_id) * block_size * head_dim
              # 跳过前面层和 head 的数据（固定偏移）
              # 例: (15*2+1)*4*64 = 31*256 = 7936
              + tl.arange(0, block_size)[:, None] * head_dim
              # 行偏移：block_size 行，每行 head_dim 个元素
              # shape: [block_size, 1] = [4, 1]
              + tl.arange(0, head_dim)[None, :])
              # 列偏移：0..head_dim-1
              # shape: [1, head_dim] = [1, 64]
    # 最终 k_ptrs 是一个 [block_size, head_dim] = [4, 64] 的指针矩阵
    # 加上 block_index * num_layers * num_kv_heads * block_size * head_dim 后
    # 就能定位到物理块 block_index 中该 layer/kv_head 的 K 数据
 
    v_ptrs = (v_cache
              + (cur_layer * num_kv_heads + my_kv_head_id) * block_size * head_dim
              + tl.arange(0, block_size)[:, None] * head_dim
              + tl.arange(0, head_dim)[None, :])
    # 结构与 k_ptrs 完全相同，指向 V Cache
 
    # --------------------------------------------------------------------------
    # 6. 初始化 online softmax 的累积变量
    # --------------------------------------------------------------------------
    # 我们用 online softmax（也叫 safe softmax）来避免数值溢出：
    #   不等所有 score 都算完再 softmax，而是边读 K/V 边维护 running max 和 sum
    #   公式: 设 m = max(scores), softmax_i = exp(score_i - m) / sum_j(exp(score_j - m))
    #
    # 用 exp2 而非 exp：
    #   score * softmax_scale，其中 softmax_scale 已预乘 log2(e)
    #   所以 exp(score * original_scale) = exp2(score * softmax_scale_with_log2e)
    #   好处：exp2 在硬件上更快，且避免 Triton 某些优化 bug（见文末注释链接）
 
    max_score = float("-1e20")   # 运行中的最大 score（用于数值稳定）
    sum_exp = 0.0                # 运行中的 exp 权重之和
    acc = tl.zeros([head_dim], dtype=tl.float32)
    # 累积的加权 V 向量，shape [head_dim]，例 [64]
 
    # --------------------------------------------------------------------------
    # 7. 分情况处理：最后一个 seq_block vs 其他 seq_block
    # --------------------------------------------------------------------------
    # 为什么要分两种情况？
    #   - 最后一个 seq_block 可能不满（例如序列长度10，第二个seq_block只有2个token）
    #     需要 masking，且不能用 tl.static_range（因为块数量运行时才知道）
    #   - 非最后 seq_block 必然是满的，可用 tl.static_range（编译时展开循环），
    #     允许 Triton 做指令重排、流水线优化，性能更好
 
    if my_start_token_idx + seq_block_size > my_seq_len:
        # ======================================================================
        # 情况 A：这是序列的最后一个 seq_block（可能不满）
        # ======================================================================
        # 例：seq_len=10, seq_block_size=8, start=8 → 8+8=16 > 10，进入此分支
        #     本 seq_block 只有 token[8:10]，2个有效 token
 
        my_num_blocks = tl.cdiv(my_seq_len - my_start_token_idx, block_size)
        # 本 seq_block 内需要处理的物理块数（向上取整）
        # 例: ceil((10-8)/4) = ceil(2/4) = 1，只需处理 1 个物理块
 
        for block_i in range(0, my_num_blocks):
            # 注意：这里用普通 range（运行时循环），因为 my_num_blocks 是运行时变量
 
            block_idx = start_block_idx + block_i
            # 逻辑块号，例: start_block_idx=2, block_i=0 → block_idx=2
 
            block_index = tl.load(
                block_table + my_seq_id * max_blocks_per_seq + block_idx
            ).to(tl.int64)
            # 查 block_table，将逻辑块号翻译为物理块号
            # 例: my_seq_id=3, max_blocks_per_seq=3
            #     block_table[3*3 + 2] = block_table[11] = 3（物理块3）
 
            k_block = tl.load(
                k_ptrs + block_index * num_layers * num_kv_heads * block_size * head_dim
            )
            # 从物理块中加载 K，shape: [block_size, head_dim] = [4, 64]
            # 物理块偏移 = block_index * num_layers * num_kv_heads * block_size * head_dim
            # 例: block_index=3, 3*32*2*4*64 = 3*16384 = 49152
            # 这把整个 k_ptrs 基址偏移到物理块3的位置
 
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1)
            # 计算 Q 与 K 的点积，shape: [block_size] = [4]
            # my_q[None, :] 变成 [1, 64]，broadcast 乘以 k_block [4, 64]，再沿 axis=1 求和
            # 相当于 [q · k0, q · k1, q · k2, q · k3]
 
            attn_score = attn_score * softmax_scale
            # 乘以缩放因子（已包含 log2(e)），为后续 exp2 做准备
 
            offs_token = (block_i * block_size         # 本块内已处理的 token 偏移
                          + my_start_token_idx         # seq_block 的起始位置
                          + tl.arange(0, block_size))  # 本块内的 token 下标
            # 例: block_i=0, my_start_token_idx=8, block_size=4
            #     offs_token = [8, 9, 10, 11]
 
            attn_score = tl.where(offs_token < my_seq_len, attn_score, float('-1e20'))
            # Masking：超出序列长度的位置设为 -inf，使其 softmax 权重为 0
            # 例: my_seq_len=10，offs_token=[8,9,10,11]
            #     token[8],[9] 有效（< 10），token[10],[11] 无效（>= 10）→ 置 -inf
 
            v_block = tl.load(
                v_ptrs + block_index * num_layers * num_kv_heads * block_size * head_dim
            )
            # 加载对应的 V，shape: [block_size, head_dim] = [4, 64]
 
            # ------------------------------------------------------------------
            # Online softmax 更新（核心公式）
            # ------------------------------------------------------------------
            cur_max_score = tl.max(attn_score, axis=0)
            # 本块内的最大 score，标量
            # 例: attn_score = [0.8, 1.2, -1e20, -1e20] → cur_max_score = 1.2
 
            new_max_score = tl.maximum(max_score, cur_max_score)
            # 全局最大 score 更新
            # 例: max_score=-1e20（初始）, cur_max_score=1.2 → new_max_score=1.2
 
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            # 计算归一化后的 exp 权重（以 2 为底）
            # 例: exp2([0.8-1.2, 1.2-1.2, -1e20-1.2, -1e20-1.2])
            #   = exp2([-0.4, 0, -inf, -inf])
            #   ≈ [0.758, 1.0, 0, 0]
            # 因为 softmax_scale 已预乘 log2(e)，所以 exp2(score*scale) = e^(score*orig_scale)
 
            old_acc_scale = tl.math.exp2(max_score - new_max_score)
            # 之前累积结果的缩放因子（因为 max_score 更新了，需要重新校正旧的 acc）
            # 例: max_score=-1e20, new_max_score=1.2 → exp2(-1e20-1.2) ≈ 0（第一次迭代）
            # 如果是第二次迭代: max_score=0.5, new_max_score=1.2 → exp2(0.5-1.2)=exp2(-0.7)≈0.616
 
            acc = acc * old_acc_scale + tl.sum(exp_attn_score[:, None] * v_block, axis=0)
            # 更新累积 V：
            #   - acc * old_acc_scale：将旧的累积结果缩放（对齐到新的 max_score 基准）
            #   - tl.sum(exp_attn_score[:, None] * v_block, axis=0)：本块的加权 V
            #     exp_attn_score[:, None] shape [4, 1]，v_block shape [4, 64]
            #     乘积 shape [4, 64]，沿 axis=0 求和得 [64]
            # 最终 acc shape: [head_dim] = [64]
 
            sum_exp = sum_exp * old_acc_scale + tl.sum(exp_attn_score, axis=0)
            # 更新累积权重之和（用于最后归一化）
            # 例: sum_exp = 0*0 + (0.758+1.0+0+0) = 1.758
 
            max_score = new_max_score
            # 保存最新的全局最大 score
 
    else:
        # ======================================================================
        # 情况 B：这 NOT 是最后一个 seq_block（满块，无需 masking）
        # ======================================================================
        # 例：seq_len=10, seq_block_size=8, start=0 → 0+8=8 ≤ 10，进入此分支
        #     本 seq_block 处理 token[0:8]，恰好 2 个满的物理块（各 4 个 token）
 
        for block_i in tl.static_range(0, seq_block_size // block_size):
            # tl.static_range：编译时已知循环次数 = seq_block_size // block_size = 8//4 = 2
            # Triton 可以展开此循环并做流水线优化，比 range() 快很多
 
            block_idx = start_block_idx + block_i
            # 例: start_block_idx=0, block_i=0 → block_idx=0
            #     start_block_idx=0, block_i=1 → block_idx=1
 
            block_index = tl.load(
                block_table + my_seq_id * max_blocks_per_seq + block_idx
            ).to(tl.int64)
            # 查 block_table
            # 例: my_seq_id=3, block_idx=0 → block_table[3*3+0]=block_table[9]=5（物理块5）
            #     my_seq_id=3, block_idx=1 → block_table[3*3+1]=block_table[10]=12（物理块12）
 
            k_block = tl.load(
                k_ptrs + block_index * num_layers * num_kv_heads * block_size * head_dim
            )
            # 加载物理块中的 K，shape: [4, 64]
            # 注意：不需要 masking，因为这个 seq_block 内所有 token 都有效
 
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1)
            # Q·K 点积，shape: [4]
            attn_score = attn_score * softmax_scale
 
            # 没有 masking 步骤（与情况 A 的区别）
 
            v_block = tl.load(
                v_ptrs + block_index * num_layers * num_kv_heads * block_size * head_dim
            )
            # 加载 V，shape: [4, 64]
 
            # Online softmax 更新（逻辑与情况 A 完全相同）
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)
            acc = acc * old_acc_scale + tl.sum(exp_attn_score[:, None] * v_block, axis=0)
            sum_exp = sum_exp * old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score
 
    # --------------------------------------------------------------------------
    # 8. 存储 Phase 1 的输出
    # --------------------------------------------------------------------------
 
    # 存储局部注意力输出（已用局部 sum_exp 归一化，但跨 seq_block 还未合并）
    offs_mid_o = (my_batch_id * num_q_heads * num_seq_blocks * head_dim
                  # 跳到第 my_batch_id 条序列
                  + my_q_head_id * num_seq_blocks * head_dim
                  # 跳到第 my_q_head_id 个 Q head
                  + my_seq_block_id * head_dim
                  # 跳到第 my_seq_block_id 个 seq_block
                  + tl.arange(0, head_dim))
                  # 取整个 head_dim 向量
    # 例: my_batch_id=0, num_q_heads=4, num_seq_blocks=2, head_dim=64
    #     my_q_head_id=2, my_seq_block_id=1
    #     offs_mid_o = 0*4*2*64 + 2*2*64 + 1*64 + [0..63] = 256+64+[0..63] = [320..383]
 
    tl.store(mid_o + offs_mid_o, acc / sum_exp)
    # 存储归一化后的局部输出，acc/sum_exp shape: [head_dim]
    # 注意：这是"局部 softmax"的输出，跨 seq_block 的 softmax 在 Phase 2 中完成
 
    # 存储 log-sum-exp（用于 Phase 2 合并时的权重计算）
    offs_mid_o_logexpsum = (my_batch_id * num_q_heads * num_seq_blocks
                            + my_q_head_id * num_seq_blocks
                            + my_seq_block_id)
    # 例: 0*4*2 + 2*2 + 1 = 0+4+1 = 5
 
    tl.store(mid_o_logexpsum + offs_mid_o_logexpsum, tl.math.log2(sum_exp) + max_score)
    # 存储 log2(sum(exp2(score_i)))，即用 log2 表示的 log-sum-exp
    # 数学推导（用 log2 体系）：
    #   sum_exp = Σ exp2(score_i - max_score)
    #   log2(sum(exp2(score_i))) = log2(sum_exp) + max_score
    # Phase 2 会用这个值来合并多个 seq_block 的结果
 
 
# ==============================================================================
# Phase 2: 跨 seq_block 合并（Online Softmax Reduction）
# ==============================================================================
 
@triton.jit
def _fwd_paged_attention_phase2(
    mid_o,          # Phase 1 的输出，shape: [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim]
    mid_o_logexpsum,# Phase 1 的 log-sum-exp，shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    o,              # 最终输出，shape: [num_decoding_seqs, num_q_heads, head_dim]
 
    decoding_seq_lens,  # shape: [num_decoding_seqs]
 
    num_q_heads:    tl.constexpr,   # 例: 4
    head_dim:       tl.constexpr,   # 例: 64
    num_seq_blocks: tl.constexpr,   # 例: 2（最大值，实际可能更少）
    seq_block_size: tl.constexpr,   # 例: 8
):
    # --------------------------------------------------------------------------
    # 1. 确定"我是谁"
    # --------------------------------------------------------------------------
    # grid = (num_decoding_seqs, num_q_heads) = (2, 4)，共 8 个 kernel
    # 每个 kernel 负责一个 (序列, Q head) 对，合并该对下所有 seq_block 的结果
 
    my_batch_id = tl.program_id(0)
    # 例: my_batch_id = 0（处理 seq_len=10 的序列）
 
    my_q_head_id = tl.program_id(1)
    # 例: my_q_head_id = 2
 
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    # 例: my_seq_len = 10
 
    my_num_seq_blocks = tl.cdiv(my_seq_len, seq_block_size)
    # 本序列实际有几个 seq_block（不是最大值，而是实际值）
    # 例: ceil(10/8) = 2
 
    # --------------------------------------------------------------------------
    # 2. 初始化累积变量
    # --------------------------------------------------------------------------
    sum_exp = 0.0
    max_score = float("-1e20")
    acc = tl.zeros([head_dim], dtype=tl.float32)
 
    # --------------------------------------------------------------------------
    # 3. 逐 seq_block 合并（Online Softmax）
    # --------------------------------------------------------------------------
    for seq_block_id in range(my_num_seq_blocks):
        # seq_block_id: 0, 1（对于 seq_len=10, seq_block_size=8）
 
        offs_mid_o = (((my_batch_id * num_q_heads + my_q_head_id) * num_seq_blocks
                       + seq_block_id) * head_dim
                      + tl.arange(0, head_dim))
        # 计算 mid_o 中本 seq_block 的偏移
        # 例: my_batch_id=0, my_q_head_id=2, num_seq_blocks=2, seq_block_id=1, head_dim=64
        #     = ((0*4+2)*2+1)*64 + [0..63] = (5)*64+[0..63] = [320..383]
 
        offs_mid_o_logexpsum = ((my_batch_id * num_q_heads + my_q_head_id) * num_seq_blocks
                                + seq_block_id)
        # 例: (0*4+2)*2+1 = 5
 
        cur_mid_o = tl.load(mid_o + offs_mid_o)
        # 加载本 seq_block 的局部归一化输出，shape: [head_dim] = [64]
 
        cur_mid_o_logexpsum = tl.load(mid_o_logexpsum + offs_mid_o_logexpsum)
        # 加载本 seq_block 的 log2(sum_exp)，标量
        # 例: 假设 seq_block_id=0 的 logexpsum=2.3，seq_block_id=1 的 logexpsum=1.1
 
        # ------------------------------------------------------------------
        # Online softmax 合并（逻辑与 Phase 1 中完全对称，但处理的是 seq_block 粒度）
        # ------------------------------------------------------------------
        new_max_score = tl.maximum(max_score, cur_mid_o_logexpsum)
        # 例（第1次迭代 seq_block_id=0）: maximum(-1e20, 2.3) = 2.3
        # 例（第2次迭代 seq_block_id=1）: maximum(2.3, 1.1) = 2.3
 
        old_scale = tl.math.exp2(max_score - new_max_score)
        # 之前累积值的缩放因子
        # 例（第1次）: exp2(-1e20 - 2.3) ≈ 0
        # 例（第2次）: exp2(2.3 - 2.3) = exp2(0) = 1.0（max 没变，所以旧结果不需要缩放）
 
        exp_score = tl.math.exp2(cur_mid_o_logexpsum - new_max_score)
        # 本 seq_block 的归一化权重
        # 例（第1次）: exp2(2.3 - 2.3) = 1.0
        # 例（第2次）: exp2(1.1 - 2.3) = exp2(-1.2) ≈ 0.435
 
        acc = acc * old_scale + exp_score * cur_mid_o
        # 更新累积输出：
        #   - acc * old_scale：校正旧的累积（对齐到新 max_score 基准）
        #   - exp_score * cur_mid_o：加入本 seq_block 的贡献
        #     注意：cur_mid_o 是已经局部归一化的，乘以 exp_score 是在"反归一化"后
        #     加权合并，这保证了全局 softmax 的正确性
        # shape: [head_dim] = [64]
 
        sum_exp = sum_exp * old_scale + exp_score
        # 更新累积权重之和
        # 例（第1次）: 0*0 + 1.0 = 1.0
        # 例（第2次）: 1.0*1.0 + 0.435 = 1.435
 
        max_score = new_max_score
 
    # --------------------------------------------------------------------------
    # 4. 存储最终输出
    # --------------------------------------------------------------------------
    offs_o = (my_batch_id * num_q_heads + my_q_head_id) * head_dim + tl.arange(0, head_dim)
    # 例: (0*4+2)*64 + [0..63] = 128+[0..63] = [128..191]
 
    tl.store(o + offs_o, (acc / sum_exp).to(tl.float16))
    # 最终归一化（除以全局 sum_exp），转换为 float16 存储
    # acc / sum_exp 就是完整序列上正确的 softmax attention 输出

# ==============================================================================
# Python 包装函数：参数准备 + Kernel 启动
# ==============================================================================
 
def paged_attention(
    q: torch.Tensor,        # [num_decoding_seqs, num_q_heads, head_dim]，例 [2, 4, 64]
    k_cache: torch.Tensor,  # [num_total_blocks, num_layers, num_kv_heads, block_size, head_dim]
    v_cache: torch.Tensor,  # 同上
    block_table: torch.Tensor,  # [num_seqs_in_engine, max_blocks_per_seq]
    model_config,           # 含 num_q_heads, num_kv_heads, head_dim, num_layers
    engine_config,          # 含 block_size, max_blocks_per_seq
    infer_state,            # 含 num_decoding_seqs, num_seq_blocks, seq_block_size,
                            #     softmax_scale, decoding_seq_lens, seq_ids, num_prefill_seqs
    cur_layer: int,         # 当前 Transformer 层号
    o: torch.Tensor         # [num_decoding_seqs, num_q_heads, head_dim]，存放输出
):
    # --------------------------------------------------------------------------
    # 1. 基本断言：确保所有张量是连续内存（Triton kernel 要求）
    # --------------------------------------------------------------------------
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_block_size % engine_config.block_size == 0
    # seq_block_size 必须是 block_size 的整数倍，确保 seq_block 边界与物理块边界对齐
    # 例: seq_block_size=8, block_size=4 → 8%4=0 ✓
    assert o.is_contiguous()

    # --------------------------------------------------------------------------
    # 2. 分配 Phase 1 的中间输出缓冲区
    # --------------------------------------------------------------------------
    mid_o = torch.empty((
        infer_state.num_decoding_seqs,  # 2
        model_config.num_q_heads,        # 4
        infer_state.num_seq_blocks,      # 2
        model_config.head_dim            # 64
    ), device=q.device, dtype=torch.float32)
    # shape: [2, 4, 2, 64]，float32 保证中间计算精度
 
    mid_o_logexpsum = torch.empty((
        infer_state.num_decoding_seqs,  # 2
        model_config.num_q_heads,        # 4
        infer_state.num_seq_blocks       # 2
    ), device=q.device, dtype=torch.float32)
    # shape: [2, 4, 2]，每个值是对应 seq_block 的 log2(sum_exp)

    # --------------------------------------------------------------------------
    # 3. 启动 Phase 1
    # --------------------------------------------------------------------------
    grid = (infer_state.num_decoding_seqs,   # 2：沿序列维度并行
            model_config.num_q_heads,         # 4：沿 Q head 维度并行
            infer_state.num_seq_blocks)        # 2：沿 seq_block 维度并行
    # 共 2×4×2=16 个 Triton kernel 实例同时运行
 
    _fwd_paged_attention_phase1[grid](
        mid_o, mid_o_logexpsum,
        q, k_cache, v_cache,
        block_table,
 
        # softmax_scale 预乘 log2(e) = 1.442695...
        # 原因：
        #   1. NVIDIA GPU 没有原生 exp 指令，exp 实际通过 x*log2(e) → exp2 实现
        #      预乘后直接用 exp2，省去每次运行时的乘法
        #   2. Triton 在循环内使用 exp 时有优化 bug（见 triton-lang/triton#2961）
        #      用 exp2 可以绕开此 bug
        # 例: original scale = 1/sqrt(64) = 0.125
        #     传入值 = 0.125 * 1.442695 ≈ 0.1803
        infer_state.softmax_scale * 1.442695040888963,
 
        infer_state.decoding_seq_lens,      # [2, 6]
 
        infer_state.seq_ids[infer_state.num_prefill_seqs:],
        # 只取解码序列的 seq_ids（跳过 prefill 序列）
        # 假设 num_prefill_seqs=1，seq_ids=[1,3,7]，则取 [3,7]
 
        infer_state.num_seq_blocks,         # 2
        cur_layer,                          # 例: 15
 
        model_config.num_layers,            # 32
        model_config.num_q_heads,           # 4
        model_config.num_kv_heads,          # 2
        model_config.num_q_heads // model_config.num_kv_heads,  # 2（GQA ratio）
        engine_config.block_size,           # 4
        model_config.head_dim,              # 64
        infer_state.seq_block_size,         # 8
        engine_config.max_blocks_per_seq,   # 3
 
        num_warps=1,
        # 每个 SM 内只用 1 个 warp（32 线程）
        # 原因：head_dim=64 或 128 时，数据量不大，更多 warp 会导致寄存器竞争
        # 对于大 head_dim（如 256），可能需要调整
 
        num_stages=4,
        # 流水线深度：4 个 stage 异步预取，隐藏内存延迟
        # Triton 会自动将下一次迭代的 load 与本次迭代的计算重叠
    )

 
    # --------------------------------------------------------------------------
    # 4. 启动 Phase 2
    # --------------------------------------------------------------------------
    grid = (infer_state.num_decoding_seqs,  # 2
            model_config.num_q_heads)        # 4
    # 共 2×4=8 个 kernel，每个合并一个 (序列, Q head) 的全部 seq_block
 
    _fwd_paged_attention_phase2[grid](
        mid_o, mid_o_logexpsum,
        o,
        infer_state.decoding_seq_lens,
        model_config.num_q_heads,       # 4
        model_config.head_dim,          # 64
        infer_state.num_seq_blocks,     # 2
        infer_state.seq_block_size,     # 8
    )
    # Phase 2 结束后，o 中存放了所有解码序列的最终 attention 输出
    # shape: [num_decoding_seqs, num_q_heads, head_dim] = [2, 4, 64]

    # from swiftllm.utils import cdiv
    # for my_batch_id in range(infer_state.num_decoding_seqs):
    #     my_q = q[my_batch_id]   # [num_q_heads, head_dim]
    #     my_block_table = block_table[infer_state.seq_ids[infer_state.num_prefill_seqs+my_batch_id]]
    #     my_num_blocks = cdiv(infer_state.decoding_seq_lens[my_batch_id], engine_config.block_size)
    #     my_k_blocks = []
    #     my_v_blocks = []
    #     for block_id in range(my_num_blocks):
    #         block_index = my_block_table[block_id]
    #         my_k_blocks.append(k_cache[block_index][cur_layer])
    #         my_v_blocks.append(v_cache[block_index][cur_layer])
    #     my_k = torch.cat(my_k_blocks, dim=1)   # [num_kv_heads, *, head_dim]
    #     my_v = torch.cat(my_v_blocks, dim=1)   # [num_kv_heads, *, head_dim]
    #     my_k = my_k.repeat_interleave(model_config.num_q_heads // model_config.num_kv_heads, dim=0)   # [num_q_heads, *, head_dim]
    #     my_v = my_v.repeat_interleave(model_config.num_q_heads // model_config.num_kv_heads, dim=0)   # [num_q_heads, *, head_dim]
    #     my_q = my_q.reshape(model_config.num_q_heads, 1, model_config.head_dim)

    #     my_q = my_q.to(torch.float32)
    #     my_k = my_k.to(torch.float32)
    #     my_v = my_v.to(torch.float32)

    #     my_attn_score = torch.bmm(my_q, my_k.transpose(1, 2)).squeeze()   # [num_q_heads, *]
    #     my_attn_score = my_attn_score * infer_state.softmax_scale
    #     # print(my_v[0])
    #     # print(my_q[0])
    #     my_attn_score = torch.where(
    #         torch.arange(my_attn_score.shape[1], device=my_attn_score.device) < infer_state.decoding_seq_lens[my_batch_id],
    #         my_attn_score,
    #         torch.full_like(my_attn_score, float('-1e20'))
    #     )
    #     # print(my_attn_score)
    #     my_attn_score = torch.softmax(my_attn_score, dim=1)   # [num_q_heads, *]
    #     my_attn_score = my_attn_score.unsqueeze(1)   # [num_q_heads, 1, *]

    #     res = torch.bmm(my_attn_score, my_v).squeeze(1)   # [num_q_heads, head_dim]
    #     o[my_batch_id] = res.reshape(-1).to(torch.float16)
