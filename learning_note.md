# Task1：Prefill 与 Attention Kernel
- 微信昵称：Sunny
- 打卡任务：llm-algo-leetcode 推理优化 | 202609 Task1 · Prefill 与 Attention Kernel
- 截止时间：2026-09-16 12:00
- 完成内容：完成最低要求 4.1
- 运行环境：魔搭社区 CPU 环境 | 预装镜像 ubuntu22.04-py312-torch2.3.1-1.40.0

## 标准 Attention 的问题
标准 Attention 在长序列下会生成一张接近 N×N 的分数矩阵，比如序列长度 4096，就会有一张 4096×4096 的大表。

因为 Softmax 需要一整行的分数都到齐才能算，所以这张大表不能算一点丢一点，必须长时间保留。如果它放在 HBM 里，还要反复读写，数据搬运很容易比计算本身更慢。

## HBM 与 SRAM
HBM 是 GPU 的大显存，容量大但慢，访问代价高；

SRAM 是片上快速缓存，容量小但快，离计算单元近。

FlashAttention 让 Q/K/V 小块和 score tile 在 SRAM 里算，中间大矩阵不写回 HBM，从而减少 HBM 读写，缓解带宽瓶颈，降低显存峰值。

## FlashAttention 的核心思想
FlashAttention 不去实际生成并保存完整的 N×N 分数矩阵，而是把计算切成小块，边算边汇总。中间的小块结果尽量留在 SRAM 里，算完就丢掉，只保留最终输出和少量统计量。

它没有减少计算量，复杂度仍然是 O(N²)，真正省的是显存占用和 HBM 数据搬运。

## FlashAttention 是如何解决 Attention 问题的
标准 Attention 是先把完整 score 矩阵算出来，再统一做 Softmax，再乘 V。

FlashAttention 改成用 tiling 把 Q、K、V 分块，一次只处理一小块，再用 online softmax 在分块过程中完成 Softmax 归约。中间 score tile 留在 SRAM，不写回 HBM，最后只把最终输出写回 HBM。这样显存峰值下降，HBM 往返减少，长序列更容易跑起来。

## Tiling 分块
Tiling 就是分块。把 Q、K、V 按序列长度切成小块，比如每块 128 个 token，每次拿一个 Q tile 和一个 K tile，算出一块 128×128 的 score tile。这块 score tile 很小，可以放在 SRAM 里算，算完就丢掉，再算下一块。

tile 不是越小越好：tile 小省 SRAM，但块数多、调度开销大；tile 大调度少，但 SRAM 压力大。

## Online Softmax
标准 Softmax 必须等一整行所有分数都到齐，才能求最大值、指数和和归一化。

Online Softmax 不用等完整一行，它边处理小块，边维护当前最大值 m 和当前指数和 l。来新块时，先更新全局最大值，如果最大值变了，就把旧的 l 和输出累加器按比例缩放，再加上新块的贡献。最后得到的结果和标准 Softmax 一样。

<img width="1240" height="848" alt="image" src="https://github.com/user-attachments/assets/1befa0ea-47bc-4697-bd99-1e6fbfdc7849" />



  
