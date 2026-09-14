# MicroDNA Map v2.0

基于 ResNet-SelfAttention 与双层滑动窗口架构的 eccDNA 识别与分析平台。

## 项目结构

```text
MicroDNA_Map/
├── config.py                  # 全局配置文件
├── requirements.txt           # Python 依赖清单
├── data/                      # 原始数据与处理后的序列目录
├── refs/                      # 参考基因组与索引目录
├── models/                    # 训练权重与日志目录
├── results/                   # 推理与评估结果目录
├── src/
│   ├── dataprocess.py         # 序列清洗与特征编码
│   ├── dataloader.py          # 数据集加载
│   ├── model.py               # 网络模型结构
│   ├── utils.py               # 通用工具函数
│   ├── hnm.py                 # 困难负例挖掘逻辑
│   └── pipeline/              # 第三方工具调用管线
├── scripts/
│   ├── process_data.py        # 提取构建数据集
│   ├── sample_negatives.py    # 基因组背景采样
│   ├── train.py               # 模型训练
│   ├── verify.py              # 模型评估
│   ├── compare_hnm.py         # HNM 效果对比
│   ├── predict.py             # 序列级推断
│   ├── batch_process.py       # FASTQ 端到端全流程
│   └── ablation_layer_size.py # 模型通道消融实验
└── benchmark/                 # 基准测试与验证脚本

```

---

## 安装

创建隔离的运行环境并安装依赖库：

```bash
conda create -n microdna python=3.9
conda activate microdna
pip install -r requirements.txt

# PyTorch 安装需与运行环境的 CUDA 版本匹配
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install seaborn cnvkit
```

运行端到端流程时，需确保以下外部依赖工具已安装并添加至系统环境变量：

* bowtie2
* bowtie2-build
* samtools
* cnvkit.py

---

## 快速使用

### 序列推断预测

使用 `predict.py` 直接对给定的 FASTA 序列文件进行打分或滑窗扫描检测。

可选参数：

* `--input` 输入的 FASTA 文件或目录路径。
* `--mode` 推断模式。`short` 用于定长序列单次分类，`long` 用于长序列滑动窗口扫描。
* `--model` 模型权重路径。
* `--limit` 判定为 eccDNA 的概率阈值，默认 0.75。
* `--batch-size` 推理批次大小，默认 256。
* `--min-region-len` 长序列模式下保留的最小预测区域长度，默认 150。
* `--no-merge` 禁用长序列模式下相邻重叠区域的自动合并。
* `--output-dir` 结果输出目录。

```bash
# 短序列模式，生成推断概率 TSV
python scripts/predict.py --input data/test_short.fa --mode short

# 长序列模式，生成候选 BED 与对应的 FASTA
python scripts/predict.py --input data/test_long.fa --mode long --limit 0.75
```

### 端到端全流程分析

使用 `batch_process.py` 从原始双端 FASTQ 测序数据开始，依次执行比对、变异检测与深度学习筛选。

可选参数：

* `--input-dir` FASTQ 文件所在目录。
* `--threads` 运行线程数。
* `--limit` 深度学习最终筛选阈值，默认 0.75。
* `--min-log2` CNVkit 过滤阈值。
* `--min-cnv-size` 与 `--max-cnv-size` 变异片段长度过滤限制。
* `--cleanup` 运行完成后删除生成的 BAM 等中间文件。

```bash
python scripts/batch_process.py --input-dir data/raw --threads 16 --cleanup
```

### 基因组直接扫描

使用 `run_microdna_map_direct.py` 直接利用模型扫描参考基因组序列以发现候选区间。

可选参数：

* `--reference` 参考基因组路径。
* `--region` 指定扫描的区域或染色体，支持多次传递。
* `--limit` 识别阈值，默认 0.75。
* `--segment_length` 内存切块大小限制。

```bash
python benchmark/detect/run_microdna_map_direct.py --reference refs/hg19.fa --region chr21 --region chr22 --limit 0.75
```

---

## 开发者指南

### 数据预处理

使用 `process_data.py` 和 `sample_negatives.py` 构建训练所需的数据集。

`process_data.py` 可选参数：

* `--input` 输入的 Excel 坐标表。
* `--label` 指定标签种类，可选 `ecc` 或 `other`。
* `--mode` 长度调整策略。可选 `expand` 向外侧扩增，`middle` 截取中间片段，`raw` 保持原样。
* `--target-len` 目标长度设置。
* `--min-span` 针对 middle 模式的跨度下限。

`sample_negatives.py` 可选参数：

* `--count` 随机采样的负例总数。
* `--ratio` 按正样本数量的比例自动计算并补充负例数量。
* `--ref` 参考基因组路径。

```bash
# 提取已知正负样本
python scripts/process_data.py --input data/raw/eccDNA_annotations.xlsx --label ecc --mode expand
python scripts/process_data.py --input data/raw/otherDNA_annotations.xlsx --label other --mode middle --min-span 600

# 扩充基因组随机背景负例
python scripts/sample_negatives.py --ratio 1.5
```

### 模型训练

使用 `train.py` 启动训练，默认包含困难负例挖掘流程。

可选参数：

* `--base-epochs` 基础训练阶段的轮数。
* `--hnm-rounds` 困难负例挖掘轮次。0 表示纯基础训练。
* `--hnm-epochs` 单个 HNM 阶段的附加训练轮数。
* `--hnm-threshold` 负例挖掘概率下限。
* `--hnm-keep-easy` 简单负例的保留比例设置。
* `--balanced` 开启类别平衡采样。
* `--layer-size` 设定模型基础通道宽度。

```bash
python scripts/train.py --base-epochs 20 --hnm-rounds 3 --hnm-epochs 15 --balanced
```

### 模型评估

使用 `verify.py` 对模型进行量化评估，输出指标报告与混淆矩阵。

可选参数：

* `--model` 模型权重路径。
* `--threshold` 预测判别阈值。
* `--no-plot` 仅输出文本指标报告，不生成相关图表。

```bash
python scripts/verify.py --model models/best_model.pth --threshold 0.5
```

### 困难负例挖掘分析

使用 `compare_hnm.py` 对比基线模型与经过 HNM 训练模型的预测概率分布变化。

可选参数：

* `--task` 运行模式。可选 `infer` 进行推理，`plot` 进行绘图，`all` 执行完整任务。
* `--model-base` 未进行 HNM 训练的基准权重路径。
* `--model-hnm` 最终模型权重路径。
* `--csv-base` 与 `--csv-hnm` 独立绘图时读取的数据路径。

```bash
python scripts/compare_hnm.py --task infer --model-base models/base.pth --model-hnm models/hnm.pth

python scripts/compare_hnm.py --task plot --csv-base results/metrics/Baseline_predictions.csv --csv-hnm results/metrics/HNM_Model_predictions.csv
```

---

## 基准测试

基准测试目录 `benchmark/` 的内部结构如下：

```text
benchmark/
├── config_real.yaml       # 真实数据 Spike-in 测试配置
├── config_random.yaml     # 随机数据 Spike-in 测试配置
├── run_benchmark.py       # Spike-in 仿真全流程评估管线
├── perturbation_test.py   # 模型特征依赖性序列扰动测试
├── models_comparison/     # 基线模型对比架构与训练代码
├── simulate/              # 仿真数据生成与测序模拟脚本
├── detect/                # 各检测软件调用封装脚本
├── evaluate/              # 重叠度计算与精确度评估工具
└── visualize/             # 评估结果绘图脚本

```

### 消融实验

**通道容量消融**
使用 `ablation_layer_size.py` 自动化测试不同的通道配置。

* 可选参数：`--sizes` 待测通道大小列表，`--epochs` 训练轮数，`--batch-size` 批次大小。

```bash
python scripts/ablation_layer_size.py --sizes 8 16 32 64 --epochs 40
```

**对比模型验证**
使用 `benchmark/models_comparison/train_eval.py` 在固定数据流下评估预置对比网络。

* 可选参数：`--model` 待评测架构，可选 `resnet50`，`resnet_no_att`，`transformer`。

```bash
python benchmark/models_comparison/train_eval.py --model resnet50 --epochs 30
```

### 序列特征扰动测试

使用 `perturbation_test.py` 验证模型对序列特征的响应，包括等 GC 含量序列打乱与单碱基突变。

可选参数：

* `--task` 运行模式，可选 `infer`，`plot`，`all`。
* `--model` 评估模型路径。
* `--input` 用于打乱与突变的真实样本集合。
* `--shuffling-n` 与 `--mutagenesis-n` 测试的序列数量限制。

```bash
python benchmark/perturbation_test.py --task all --model models/best_model.pth --input data/processed/eccDNA.fa
```

### Spike-in 仿真实验

使用 `run_benchmark.py` 执行统一的 Spike-in 数据模拟、工具检测调用与重叠率评估全流程管线。

可选参数：

* `--config` 指定 YAML 格式的实验配置定义文件。
* `--phase` 设定运行阶段编号，数字逗号分隔。未指定则依次运行全部阶段。
* `--quick` 覆盖配置，启动限制检测区域的快速测试模式。

```bash
python benchmark/run_benchmark.py --config benchmark/config_real.yaml --phase 1,2,3,4,5,6,7,8,9
```