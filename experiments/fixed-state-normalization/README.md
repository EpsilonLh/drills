# 固定尺度状态归一化消融

固定 i2c、种子 0/1/2、100×10。原训练/冻结/均匀随机复用已核验历史证据；新增固定归一化训练/冻结各三次。第100轮模型使用30000–30029评估，均匀随机只运行一次，共390条实际轨迹。

`protocol.yml` 在运行前锁定；不根据结果换种子、调公式或选择中间模型。门槛为旧训练减新训练平均至少1 LUT，且至少两个种子改善。分别报告工程收益及冻结减训练的学习收益。

在仓库根目录执行：

```bash
.tools/conda-env/bin/python -B experiments/fixed-state-normalization/check.py
.tools/conda-env/bin/python -B -u experiments/fixed-state-normalization/run.py all --plot-python /你的绘图Python路径
```

绘图Python需要ReportLab与PyMuPDF；Codex内置文档运行环境可直接使用，不改变训练依赖。原始输出位于`results/fixed-state-normalization/`。完整结果不能覆盖；中断后的显式恢复使用`train --resume`或`evaluate --resume`，指纹改变会拒绝恢复。失败不自动重复。

旧输入文件由`historical-inputs.json`固定哈希。所有历史文件保持不变；复用记录复制到本次结果目录。新旧模式使用同一生产归一化实现，检查点记录模式与特征顺序。

交付报告为`report.md`；压缩证据包含初始/最终模型、候选日志、全部评估记录和最佳网表，内部`SHA256.json`可逐文件校验。中间50轮模型仅保留本机，正式比较始终使用100轮模型。

交付文档或视觉核对记录更新后，使用 `sh experiments/fixed-state-normalization/package.sh` 再打包。此入口会先把上一次生成的哈希清单移到临时备份，避免将清单包含进自身，不修改实验源码或搜索数据。首次完整运行仍使用上述 `all` 命令。
