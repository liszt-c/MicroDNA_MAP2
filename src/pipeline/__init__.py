"""
src/pipeline/__init__.py - 管线模块与统一工厂函数
"""
from typing import Any
from .cnvkit_pipeline import CNVKitPipeline
from .micro_coverage_pipeline import MicroCoveragePipeline


def get_pipeline(pipeline_name: str = "micro_coverage", output_dir=None, ref_genome=None, **kwargs) -> Any:
    """
    根据 pipeline_name 返回对应的分析流程实例 (策略模式解耦)
    :param pipeline_name: 'micro_coverage' (默认推荐) 或 'cnvkit' (传统大尺度流程)
    """
    p_name = str(pipeline_name).lower().strip()
    if p_name in ("micro_coverage", "microcov", "coverage", "micro"):
        return MicroCoveragePipeline(output_dir=output_dir, ref_genome=ref_genome, **kwargs)
    elif p_name in ("cnvkit", "cnv"):
        return CNVKitPipeline(output_dir=output_dir, ref_genome=ref_genome)
    else:
        raise ValueError(
            f"Unknown pipeline: '{pipeline_name}'. Valid options: 'micro_coverage', 'cnvkit'."
        )


__all__ = ["CNVKitPipeline", "MicroCoveragePipeline", "get_pipeline"]