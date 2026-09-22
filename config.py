"""
config.py - 全局配置中心
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 参考基因组
REFS_DIR = PROJECT_ROOT / "refs"
HG19_FA = REFS_DIR / "hg19.fa"
# 可选: CNVkit 参考 profile (.cnn)。存在时 batch 使用 -r，否则退化为 flat reference
CNVKIT_REF_CNN = REFS_DIR / "cnvkit_ref.cnn"

# 模型与结果目录
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
CNVKIT_TEMP_DIR = RESULTS_DIR / "cnvkit_temp"
MICRO_COVERAGE_TEMP_DIR = RESULTS_DIR / "micro_coverage_temp"

# 确保目录存在 (exist_ok=True 在多进程下安全)
for _dir in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REFS_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
    CNVKIT_TEMP_DIR,
    MICRO_COVERAGE_TEMP_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# 默认超参数
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_STEP_SIZE = 5            # 学习率衰减间隔 (epoch)
DEFAULT_GAMMA = 0.6              # 学习率衰减率
DEFAULT_FLOODING_B = 0.0001      # Flooding 正则化参数, 0 表示关闭

# ---- 困难负例挖掘 (HNM) 多阶段参数 ----
DEFAULT_BASE_EPOCHS = 30         # 基础训练阶段的 Epoch 数
DEFAULT_HNM_ROUNDS = 1           # 困难负例挖掘的轮数 (0表示不进行挖掘)
DEFAULT_HNM_EPOCHS = 20          # 每一轮 HNM 附加训练的 Epoch 数
DEFAULT_HNM_THRESHOLD = 0.1      # 判断为困难负例的概率阈值
DEFAULT_HNM_KEEP_EASY = 0.1      # 保留的简单负例比例 (防止灾难性遗忘)

SEQUENCE_LENGTH = 400
NUM_CLASSES = 2
RANDOM_SEED = 42
NUM_WORKERS = 4

# 模型结构参数 (消融实验可覆盖)
LAYER_SIZE = 8                   # ResNet 基础通道数, ablation_layer_size.py 可通过 --layer-size 覆盖

# 滑动窗口推理参数 (与原版 run.py 保持一致)
SLIDE_STEP1 = 10        # 第一次滑动步长 (bp)
SLIDE_WINDOW2 = 10      # 第二次滑动窗口大小 (窗口数)
SLIDE_STEP2 = 3         # 第二次滑动步长 (窗口数)

# 微尺度局部覆盖度分析参数 (针对 microDNA 物理尺度设计)
DEFAULT_MICRO_WINDOW_SIZE = 200      # 微窗口大小 (bp)
DEFAULT_MICRO_FOLD_CHANGE = 1.3      # 局部富集判定倍数
DEFAULT_MICRO_MIN_LEN = 150          # 单分子长度下限 (bp)
DEFAULT_MICRO_MAX_LEN = 1000         # 单分子长度典型上限 (bp)
DEFAULT_MICRO_CLUSTER_MAX_LEN = 5000 # 宽富集簇 (Cluster) 容忍上限 (bp)

# 外部工具路径 (确保在 PATH 中, 或在此改为绝对路径)
BOWTIE2 = "bowtie2"
BOWTIE2_BUILD = "bowtie2-build"
SAMTOOLS = "samtools"
CNVKIT = "cnvkit.py"