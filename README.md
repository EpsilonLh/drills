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
  output_dir: results
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

结果默认保存到 `results/`：`results.md` 查看各个种子的 LUTs、levels 和训练耗时；各电路的 `seed-*` 目录保存检查点、最优网表和逐步日志。

## 新增映射性能状态实验

`params-new-state.yml` 在原九维结构状态之后增加两维：当前映射 LUT 数相对第 0 步映射的比例，以及 `(max_levels - 当前映射 levels) / max_levels`。原九维继续按每回合均值、方差标准化；新增两维直接输入 Actor 和 Critic，保留零余量表示约束边界的含义。它们复用已有映射结果，不增加工具调用。

```bash
# 使用本项目已有的本地环境运行改进版
.tools/conda-env/bin/python drills.py train fpga params-new-state.yml

# 检查状态语义、原版兼容性、工具调用次数和续训
.tools/conda-env/bin/python -m unittest discover -s tests -v

# 与之前保存在 results/ 的原版结果比较
.tools/conda-env/bin/python compare_new_state.py
```

新结果保存在 `results/new-state/`，对比说明见 [new-state-comparison.md](new-state-comparison.md)。重跑实验需指定新的输出目录，已存在的训练不会被覆盖。此配置中的工具路径指向本机 `.tools/conda-env/`，在其他机器运行时需调整。

不配置 `method.performance_features`，或将其设为空列表 `[]`，继续使用原九维状态。新增特征时必须从头训练，不能使用原九维检查点；同一改进版可用 `--resume` 续训，特征定义必须匹配。

### 同维度对照与十种子验证

`method.performance_feature_mask: [0, 0]` 将新增两维置零，但保持十一维网络；`[1, 1]` 保留性能信息。默认不配置掩码时，两维均保留。这样可用相同初始权重比较新增信息的作用。掩码是检查点定义的一部分，续训不能改变。

```bash
# 第一步：种子 0/1/2 的十一维零特征对照
.tools/conda-env/bin/python -B run_state_validation.py first
.tools/conda-env/bin/python -B compare_state_validation.py first

# 第二步：两组扩展至预定种子 0–9，复用已完成种子
.tools/conda-env/bin/python -B run_state_validation.py ten
.tools/conda-env/bin/python -B compare_state_validation.py ten
```

原结果保持在 `results/` 和 `results/new-state/`；新对照写入 `results/state-validation/`。各阶段命令、代码快照、初始化核验和进程状态一并保存，每个训练命令有 60 分钟硬超时。阶段结果分别见 `state-validation-first.md` 和 `state-validation-ten.md`，配套 CSV 与 JSON 包含逐种子数据和分析设置。

## 参考基线

采用 LUT6 映射；参考运行使用 100 episodes × 10 iterations、种子 0/1/2。参考结果取三个种子各自训练搜索最优可行解的均值。

| 电路 | resyn2 levels | resyn2 LUTs | 参考结果 levels | 参考结果 LUTs |
|---|---:|---:|---:|---:|
| int2float | 3 | 48 | 3 | 44.00 |
| i2c | 4 | 322 | 4 | 303.67 |
| max | 41 | 777 | 41 | 772.33 |
