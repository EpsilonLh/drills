# 200轮 × 5步受控实验

该套件只改变原实验的轮数和每轮动作数。学习率0.001、奖励、网络、归一化、Adam、动作集合和约束均保持原设置。

训练：trained / frozen / uniform × int2float / i2c / max × seed 0/1/2，共27次200×5运行。
保存第0/100/200轮模型；第200轮为主结果。新评估种子20000–20029，每条序列从原电路开始执行5步，全程停止学习。
初始化与冻结经不变性验证后共享评估；uniform每个电路只执行一份30条基线。因此有900条实际评估序列、1,080条逻辑配对行。

## 复现

在仓库根目录使用已安装requirements.txt的Python环境，配置protocol.yml中ABC和Yosys的路径。协议中的相对路径以该YAML所在目录为基准。

```bash
.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/check.py
.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-200x5/run.py
.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-200x5/evaluate.py
.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/verify.py
.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/evaluate.py --report-only
# 使用安装了ReportLab与PyMuPDF的Python运行plot.py
python experiments/learning-effectiveness-200x5/plot.py
.tools/conda-env/bin/python -B experiments/learning-effectiveness-200x5/package.py
```

首轮执行使用干净的结果目录。训练/评估中断后，在完全相同源码、协议、工具和输入指纹下分别使用run.py --resume、evaluate.py --resume。
不要用旧设置的检查点继续训练成新方案。修改协议或源码时需使用新的干净结果目录，并重新生成测试及指纹记录。
report-only只重读已有证据，不重新执行ABC或训练。

config()使用仓库原算法提交adf7bef的配置进行一致性检查，因此需要包含该提交的完整Git历史。
训练、评估及报告生成不依赖旧实验的本地模型；若历史trajectories.csv缺失，只省去旧100×10比较。
preserved-inputs.json记录此次交付机器的旧证据哈希，package.py用于该机器的保留检查；其他机器新跑时应先重建自己的保留清单。
具体命令为 `python experiments/learning-effectiveness-200x5/package.py --capture-preservation`，在首次运行新实验前执行一次。
已有运行过程中不要重建该清单，否则会失去对旧文件变更的检测依据。

## 交付物与判断

report.md为中文报告；training.csv、evaluation.csv、comparisons.csv等包含全部种子、缺失和不可行结果。
search-curves.svg/png为训练搜索曲线，policy-evaluation.svg/png为固定策略的检查点评估。
tests.log、test-results.json、log-validation.json、netlist-validation.json记录验证；evidence.tar.gz内含SHA256.json。
artifact-manifest.json核对交付物，model-manifest.json核对未纳入Git的本地权重。
原始输出和模型位于results/learning-effectiveness-200x5/，固定模型重放需要复制这些权重或重新训练。

主结论比较第200轮对初始化和uniform的5步表现，优先报告可行率；全部配对可行时才汇总LUT收益。
只在两种基线的主指标探索性区间均支持改善、且三个训练种子方向一致时报告一致收益。
训练累计最好值仅为辅助指标。相同1,000动作预算下，新旧实验仍同时改变序列长度和更新频率，不作单因素因果解释。

运行监督采用3个workers、每进程1个Torch线程、每分钟进度记录、单任务30分钟硬超时。
失败和未完成任务保留证据；不以中期模型替代缺失的第200轮主模型。
