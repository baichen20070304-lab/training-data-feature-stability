# 原始数据获取与复现

公开版本不含原始 XLSX。原仓库已查看的根目录未见明确许可证，因此本次发布没有打包原始数据、原论文 PDF 或复制的上游 Python 文件。此处保留来源、固定版本与校验方法，不对数据使用权作额外结论。

## 获取原始数据

1. 打开原仓库固定版本的 [Science_feature_data.xlsx](https://github.com/Andy-lucky2005/feature_importance_analysis/blob/c17c6832e5d18d66f059ba863375161490775f96/Feature_analysis_MSI/MSI_dataset/Science_feature_data.xlsx)，使用页面的原始文件下载按钮取得 XLSX。
2. 在本项目根目录的 `data/` 下保存为 `Science_feature_data.xlsx`，即 `data/Science_feature_data.xlsx`。
3. 从项目根目录校验文件 SHA-256：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'data/Science_feature_data.xlsx'
```

预期值（大小写不影响比较）：

```text
37e36004645d1a36b36f4cea3b22ed00d0c127bf675e564a8227b26da88a0550
```

固定上游提交为 `c17c6832e5d18d66f059ba863375161490775f96`；上游路径为 `Feature_analysis_MSI/MSI_dataset/Science_feature_data.xlsx`。来源及文件哈希见 [provenance/source.json](../provenance/source.json)。其中列出的两个上游 Python 文件是来源记录，未随本公开版本分发。

数据用于 MSI 黏附能预测：178 条唯一记录，14 个主描述符，目标单位 J/m²；文件中的次级特征不纳入本实验。

## 不下载数据也能查看结果

从项目根目录打开 [analysis.ipynb](../analysis.ipynb)，选择项目 `.venv` 的 Python 内核并运行全部单元格。该 notebook 读取保存的 CSV、`completion.json`、PNG 和中文报告，不读取 XLSX 或训练检查点。[中文报告](../results/full/report_zh.md) 也可直接阅读。

Windows PowerShell 中安装环境，并用命令执行保存结果的 notebook：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv\Scripts\python.exe -m ipykernel install --user --name feature-stability --display-name 'Python (.venv feature-stability)'
@'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('analysis.ipynb', as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='feature-stability',
                        resources={'metadata': {'path': str(Path.cwd())}})
client.execute()
nbformat.write(nb, 'analysis_view.ipynb')
print('Saved analysis_view.ipynb')
'@ | & .\.venv\Scripts\python.exe
```

运行目录应为项目根目录；生成的 `analysis_view.ipynb` 是本地重新执行的副本。Python 内核安装命令为此虚拟环境注册一个名称；如果使用编辑器直接选择 `.venv`，可省略注册与上述命令执行步骤。

## 从原始数据重新运行

取得并校验数据后，从项目根目录运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -u run_experiment.py --config docs/reproduce.yaml --stage all
```

部分测试依赖 XLSX，应在下载数据后运行完整测试。Python 要求 ≥3.11，本次保存结果使用 Python 3.12；优先使用锁定依赖。Linux/macOS 将解释器路径替换为 `.venv/bin/python`。

[docs/reproduce.yaml](../docs/reproduce.yaml) 保持原科学配置并将输出写入 `results/reproduction/`。运行包括试跑、512/1024 KernelSHAP 预算校准、完整训练与分析；完整实验为 364 次拟合。新结果位于 `results/reproduction/full/`，根目录 `analysis.ipynb` 会重新生成并指向这批新结果。

公开版本没有完整实验的 364 个逐次检查点，也没有试跑检查点。现有 CLI 的 `--stage analysis` 会校验原始数据哈希并读取完整检查点，不能从公开 CSV 独立重建图表。要重建全部结果，请先执行上述 `--stage all`；之后可使用同一配置的 `--stage analysis` 重建新输出。

[返回项目说明](../README.md)
