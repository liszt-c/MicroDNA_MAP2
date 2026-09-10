# MicroDNA Map v2.0

基于 ResNet-SelfAttention 的 eccDNA 识别与分析平台。  
统一配置管理、合并 FASTA 存储、标准化训练/推理接口、完整 CNVkit 流程集成。

## 项目结构

```text
MicroDNA_Map/
├── config.py                  # 全局配置中心
├── requirements.txt           # Python 依赖
├── data/
│   ├── raw/                   # 原始 Excel / FASTQ
│   └── processed/             # 合并后的 eccDNA.fa / otherDNA.fa
├── refs/                      # 参考基因组 (hg19.fa + .fai)
├── models/                    # 训练权重 (best_model.pth / last_model.pth)
├── results/
│   ├── predictions/           # 推理结果 (BED / FASTA / TSV)
│   ├── metrics/               # 评估报告与图表
│   └── cnvkit_temp/           # CNVkit 中间文件
├── src/
│   ├── __init__.py
│   ├── dataprocess.py         # 序列清洗 + One-hot 编码 + Header 解析
│   ├── dataloader.py          # PyTorch Dataset (字节偏移索引, 惰性加载)
│   ├── model.py               # ResNet-SelfAttention (1D)
│   ├── utils.py               # Logger / subprocess / samtools 批量提取
│   └── pipeline/
│       ├── __init__.py
│       └── cnvkit_pipeline.py # CNVkit 全流程封装
└── scripts/
    ├── process_data.py        # Excel -> 合并 FASTA
    ├── train.py               # 模型训练
    ├── verify.py              # 模型评估
    ├── predict.py             # 推理 (short / long)
    └── batch_process.py       # FASTQ -> MicroDNA 全流程
```

## 安装

```bash
conda create -n microdna python==3.9
conda activate microdna
pip install -r requirements.txt

# PyTorch (根据 CUDA 版本选择)
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# 外部工具 (需在 PATH 中或在 config.py 中指定绝对路径)
# bowtie2, samtools, cnvkit
pip install cnvkit
```

## 快速开始

### 1. 数据预处理

将标注 Excel 放入 `data/raw/`, 参考基因组放入 `refs/hg19.fa`:

```bash
# eccDNA (短区域向两侧扩展取真实侧翼)
python scripts/process_data.py \
    --input data/raw/eccDNA_annotations.xlsx \
    --label ecc --mode expand --ref refs/hg19.fa

# otherDNA (长跨度区域取正中间 400bp)
python scripts/process_data.py \
    --input data/raw/otherDNA_annotations.xlsx \
    --label other --mode middle --min-span 600 --ref refs/hg19.fa
```

> **注意**: 若 ecc / other 使用不同参考基因组 (如 hg19 vs GRCh38),  
> 请分别准备并通过 `--ref` 指定。

### 2. 训练

```bash
python scripts/train.py --epochs 50 --batch-size 256 --balanced
```

可选参数: `--lr`, `--weight-decay`, `--step-size`, `--gamma`,  
`--flooding-b`, `--val-split`, `--seed`, `--output-dir`

### 3. 评估

```bash
python scripts/verify.py --threshold 0.5
```

输出: `results/metrics/evaluation_report_threshold0.5.txt`  
图表: `results/metrics/evaluation_metrics_threshold0.5.png`

### 4. 推理

```bash
# 长序列滑窗识别
python scripts/predict.py --model ./models/6.pth --input path/to/seqs.fa --mode long

# 短序列分类
python scripts/predict.py --model ./models/6.pth --input path/to/short.fa --mode short
```

### 5. FASTQ 全流程

将配对 FASTQ 放入 `data/raw/`:

```bash
python scripts/batch_process.py --threads 16 --cleanup
```

流程: bowtie2 比对 → samtools sort/index → cnvkit batch+call →  
候选区域提取 → 滑窗预测 → BED + FASTA 输出

## 标签约定

| 类别 | 数值 | 说明 |
|------|------|------|
| otherDNA | 0 | 基因组背景 / 非 eccDNA |
| eccDNA | 1 | 染色体外环状 DNA |