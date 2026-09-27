# Git交付与本地证据包

200×5的完整证据包为169,319,913字节，超过GitHub普通Git文件的100 MiB限制。按用户选择，该压缩包保留在本地，Git交付代码、配置、报告、CSV、图表、验证记录和哈希清单。

本地证据包位置：`experiments/learning-effectiveness-200x5/evidence.tar.gz`。

SHA-256：`782fbb93ebf085d45f09901fc0fdcce685b6e50ee9d5ffe1778fadaabbb336d2`。

实验打包脚本仍会在本地生成和校验该文件；其内容及原哈希清单不变。仅克隆Git仓库不会获得该完整压缩包，跨机器交付需要另行复制它，训练权重也按原方案保留在本地结果目录。

本次只从两个未推送提交的树中移除了证据压缩包，其他文件完全一致，原代码提交`e410b19`保持不变。提交号映射：

- `9a405b8` → `ab90c60`
- `2603ae4` → `9a9dae2`

远端已有历史未改写。修复后使用普通推送：

```bash
git push origin learning-effectiveness
```

修复前的三个本地提交保存在`results/git-push-repair-20260928/before.bundle`；该bundle依赖仓库已有的`8563a1d`提交。校验与旧新提交映射保存在同目录JSON记录。该恢复备份不纳入Git交付。

GitHub文件大小规则：[官方文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
