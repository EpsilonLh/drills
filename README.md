# DRiLLS

基于 [官方 DRiLLS](https://github.com/scale-lab/DRiLLS) 的 FPGA 版本，使用 PyTorch CPU 训练 A2C，搜索满足深度约束且 LUT 数较少的 ABC 综合序列。

## 环境

- Linux（已在 Ubuntu 20.04 x86_64 验证），无需 GPU。
- Python 3.9。
- PyTorch 2.8.0+cpu、NumPy、PyYAML、filelock，由 `requirements.txt` 安装。
- [ABC](https://github.com/berkeley-abc/abc) 和 [Yosys](https://github.com/YosysHQ/yosys)（本次实验使用 ABC 1.01、Yosys 0.44；安装方法见各仓库 README）。

使用任意 Python 3.9 环境（如 Conda、venv 或已有环境），在项目目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

ABC 和 Yosys 的安装方式及编译依赖见上面的官方仓库链接。安装完成后配置可执行文件路径。

在 `params.yml` 中设置工具路径：

```yaml
runtime:
  abc_binary: /你的工具目录/bin/yosys-abc
  yosys_binary: /你的工具目录/bin/yosys
  output_dir: results/change-rewards
  workers: 3
```

如果工具已在 PATH 中，可直接填写 `yosys-abc` 和 `yosys`。

## 配置与运行

所有实验参数均在 `params.yml` 中，可自行修改。默认运行 int2float、i2c、max，使用种子 0、1、2，每次训练 100 轮、每轮 10 步。输入电路已包含在 `benchmarks/` 中。

```bash
# 运行 resyn2 基线和全部训练搜索
python drills.py train fpga

# 续训至 params.yml 中设置的总轮数
python drills.py train fpga --resume

# 仅运行 resyn2 基线
python drills.py baseline fpga

# 重新生成结果汇总
python drills.py report fpga
```

使用其他配置时，在 `fpga` 后追加 YAML 路径。相对路径以配置文件所在目录为基准。

结果默认保存到 `results/change-rewards/`：`results.md` 查看各个种子的 LUTs、levels 和训练耗时；各电路的 `seed-*` 目录保存检查点、最优网表和逐步日志。原有 `results/` 数据保留作为旧奖励的历史对照。

## LUT 幅度奖励

本分支默认配置为：

```yaml
method:
  reward:
    feasible_mode: normalized_delta
    feasible_scale: 1.0
```

仅当前后两步都满足 `levels <= max_levels` 时，奖励为
`feasible_scale * (previous LUTs - current LUTs) / initial LUTs`。
`initial LUTs` 是每轮初始序列映射出的 step 0 LUT 数，整轮固定。
改善和恶化按相同尺度计分；LUT 数相同时奖励为零。新模式拒绝初始 LUT 为零的电路。
跨越约束和目标状态不可行时仍使用原有奖励表。

`feasible_mode: table` 使用原奖励；旧配置省略新增字段时也使用原奖励。
`feasible_scale` 缺省为 1.0，必须为有限正数。
训练仍标准化每轮折扣回报，因此尺度在全程可行的回合中通常会被抵消，混合奖励分支时仍会影响训练。
`49→50→49` 的新原始奖励合计为零；折扣回报不保证相消。

实验必须从头训练到新的输出目录，不能以旧奖励的模型续训作为新策略对照。
同配置新策略可使用 `--resume` 接续中断的训练。
本次固定 c=1、三个电路及种子 0/1/2 的代码、命令和实验证据见
[实验报告](experiments/change-rewards/report.md)。

## i2c 回报处理对照

`method.learning_enabled` 缺省为 `true`。设为 `false` 时跳过网络与 Adam 更新，
保留初始策略、状态处理和随机采样，用于同预算的冻结策略对照。
`normalization.returns` 仅接受 `standardize` 或 `none`；后者保留原始折扣回报。
`returns_epsilon` 必须为有限正数。默认配置继续使用逐轮回报标准化。

本次仅测试 i2c，复用已有的 standardize/c=1 结果，新增 none/c=1、none/c=10
和 none/c=1 的冻结组。每组采用种子 0/1/2、100 轮 × 10 步。

```bash
.tools/conda-env/bin/python -B -u experiments/i2c-return-handling/run.py
# 中断后使用原配置、原源码接续
.tools/conda-env/bin/python -B -u experiments/i2c-return-handling/run.py --resume
.tools/conda-env/bin/python -B experiments/i2c-return-handling/evaluate.py
```

首次运行要求输出目录尚不存在；复核本地已完成结果时仅运行 evaluate.py。
运行器和检查点使用实验指纹拒绝跨组或更改配置后的续跑。
结果保存在 `results/i2c-return-handling/`，代码与紧凑证据见
[i2c 实验报告](experiments/i2c-return-handling/report.md)。

## 参考基线

采用 LUT6 映射；参考运行使用 100 episodes × 10 iterations、种子 0/1/2。参考结果取三个种子各自训练搜索最优可行解的均值。

| 电路 | resyn2 levels | resyn2 LUTs | 参考结果 levels | 参考结果 LUTs |
|---|---:|---:|---:|---:|
| int2float | 3 | 48 | 3 | 44.00 |
| i2c | 4 | 322 | 4 | 303.67 |
| max | 41 | 777 | 41 | 772.33 |
