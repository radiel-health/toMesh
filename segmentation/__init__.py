"""Segmentation package: TotalSegmentator wrapper + mask post-processing."""

from .segment import run_segmentation
from .postprocess import clean_mask

__all__ = ["run_segmentation", "clean_mask"]
