# MicroDNA Map v2.0

基于 ResNet-SelfAttention 架构的 eccDNA (MicroDNA) 识别与分析平台。

## 项目结构

```text
MicroDNA_Map/
├── config.py                  # 全局配置中心 (超参数、路径、工具链、多阶段 HNM 配置)
├── requirements.txt           # Python 依赖清单
├── data/
│   ├── raw/                   # 原始标注 Excel / 双端 FASTQ 文件
│   └── processed/             # 处理并合并后的模型输入 (eccDNA.fa / otherDNA.fa)
├── refs/                      # 参考基因组目录 (需包含 hg19.fa 及其索引)
├── models/                    # 训练输出目录 (保存权重文件及 TensorBoard 日志)
├── results/
│   ├── predictions/           # 推理结果目录 (BED / FASTA / TSV)
│   ├── metrics/               # 评估报告与可视化图表
│   └── cnvkit_temp/           # 全流程中产生的 CNVkit 与比对中间文件
├── src/
│   ├── __init__.py
│   ├── dataprocess.py         # 序列清洗、One-hot 编码、Header 解析
│   ├── dataloader.py          # 高效 PyTorch Dataset
│   ├── model.py               # 核心网络: ResNet-SelfAttention (1D)
│   ├── utils.py               # 进程调度、samtools 序列提取等工具
│   ├── hnm.py                 # 在线困难负例挖掘 (Online Hard Negative Mining) 核心逻辑
│   └── pipeline/
│       └── cnvkit_pipeline.py # FASTQ -> BAM -> CNV 提取自动化
└── scripts/
    ├── process_data.py        # 从 Excel 提取序列并转为 FASTA
    ├── sample_negatives.py    # (离线挖掘) 全基因组随机采样背景片段
    ├── train.py               # 模型训练脚本 (集成多阶段在线困难负例挖掘与类别平衡)
    ├── ablation_layer_size.py # 自动化模型通道数 (LAYER_SIZE) 消融实验
    ├── verify.py              # 模型验证脚本 (分类报告、混淆矩阵、ROC 曲线)
    ├── compare_hnm.py         # 对比分析脚本 (解耦的 CSV 推理与 KDE 密度图绘制)
    ├── predict.py             # 推理脚本 (短序列直出与长序列滑窗重叠扫描)
    └── batch_process.py       # 端到端全流程脚本 (FASTQ -> CNV -> MicroDNA)

```

## 环境与依赖安装

推荐使用 Conda 创建隔离的运行环境：

```bash
conda create -n microdna python=3.9
conda activate microdna
pip install -r requirements.txt

# PyTorch 安装 (请依据硬件 CUDA 版本调整)
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装绘图与 CNVkit
pip install seaborn cnvkit

```

**外部工具链要求:**
运行端到端流程及数据提取时，需确保以下生信工具在系统 `PATH` 中，或在 `config.py` 中指定绝对路径：

* `bowtie2` 及 `bowtie2-build`
* `samtools`

---

## 详细使用指南

### 1. 数据准备与双重负例挖掘

#### 1.1 提取确切标注序列 (`process_data.py`)

提取 Excel 中的 eccDNA (正例) 与明确的 otherDNA (负例)。

```bash
# 提取正例 (eccDNA)
python scripts/process_data.py --input data/raw/eccDNA_annotations.xlsx --label ecc --mode expand --ref refs/hg19.fa

# 提取已有负例 (otherDNA)
python scripts/process_data.py --input data/raw/otherDNA_annotations.xlsx --label other --mode middle --min-span 600 --ref refs/hg19.fa

```

#### 1.2 离线全基因组背景覆盖 (`sample_negatives.py`)

自动剔除包含 N 的低质量区域（着丝粒、拼接缝等），以产生海量干净的基因组背景片段。

```bash
# 自动比例模式：计算并补充随机背景，直到负例总数为正例的 1.5 倍
python scripts/sample_negatives.py --ratio 1.5 --ref refs/hg19.fa

```

---

### 2. 模型训练与多阶段困难负例挖掘 (Multi-round HNM)

在实际训练中，我们采用**多阶段级联挖掘策略**：首先进行 Base 训练（如 20 个 Epoch）；随后每进入一个新挖掘阶段，模型会自动加载前一阶段的 Best 模型权重，筛选出迷惑性最大的困难负例，并重置优化器再训练 15 个 Epoch。这种机制确保了模型能够逐层逼近最精确的分类边界。

```bash
# 基础训练 20 个 Epoch，再进行 3 轮 HNM（附加 15 个 Epoch），总计 65 Epoch
python scripts/train.py --base-epochs 20 --hnm-rounds 3 --hnm-epochs 15 --batch-size 256 --balanced

```

**HNM 专属参数:**

* `--base-epochs`: 基础阶段训练轮数 (默认 30)。
* `--hnm-rounds`: 进行 HNM 挖掘的轮次，0 表示纯基础训练 (默认 1)。
* `--hnm-epochs`: 每一轮 HNM 挖掘后附加的独立训练轮数 (默认 20)。
* `--hnm-threshold`: 将被保留进行重点学习的困难负例概率底线 (默认 0.1)。
* `--hnm-keep-easy`: 要保留的简单负例比例，维持全基因组背景分布不致灾难性遗忘 (默认 0.1)。

---

### 3. 困难负例挖掘效果验证 (`compare_hnm.py`)

采用低耦合高内聚设计，支持独立执行长耗时推断（输出独立 CSV）与灵活的轻量级作图（包含 2x2 核密度估计对比图）。

**方式一：一次性执行推断与制图**

```bash
python scripts/compare_hnm.py \
    --task all \
    --model-base models/baseline_model.pth \
    --model-hnm models/best_model.pth

```

**方式二：分离式执行 (推荐)**
阶段 1: 仅执行推断并持久化保存 CSV, 便于其他统计软件复用排查顽固假阳性。

```bash
python scripts/compare_hnm.py \
    --task infer \
    --model-base models/baseline_model.pth \
    --model-hnm models/best_model.pth

```

阶段 2: 修改绘图参数后，无需重新推理，直接读取 CSV 高速生成 PDF 报告。

```bash
python scripts/compare_hnm.py \
    --task plot \
    --csv-base results/metrics/hnm_comparison/Baseline_predictions.csv \
    --csv-hnm results/metrics/hnm_comparison/HNM_Model_predictions.csv

```

---

### 4. 序列级分类推断 (`predict.py`)

内存友好的流式推断程序。

```bash
# 短序列模式 (定长短序列批量快速打分，生成 TSV 报表)
python scripts/predict.py --input data/test_short.fa --mode short

# 长序列模式 (未知长序列滑窗预测，支持重叠融合，输出标准 BED 与 FASTA)
python scripts/predict.py --input data/test_long.fa --mode long --limit 0.99

```

---

### 5. 端到端生信全流程挖掘 (`batch_process.py`)

支持从测序下机数据到深度学习目标片段识别的一体化工作流。

```bash
python scripts/batch_process.py --input-dir data/raw --threads 16 --cleanup

```

---
### 6. 模型架构消融与对比基准

```bash
# 训练并评估标准 ResNet50 基线
python benchmark/models_comparison/train_eval.py --model resnet50 --epochs 30

# 训练并评估无注意力机制的 ResNet (Ablation)
python benchmark/models_comparison/train_eval.py --model resnet_no_att --epochs 30

# 训练并评估 Transformer 架构 (在 SI 补充材料中论证其在序列数据上的局限性)
python benchmark/models_comparison/train_eval.py --model transformer --epochs 30
```

