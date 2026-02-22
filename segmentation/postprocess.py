"""
Mask post-processing utilities.

Cleans a binary segmentation mask with:
  - Small-object removal (scipy label + size filter)
  - Hole filling (binary_fill_holes per 2D slice)
  - Binary dilation for surface continuity
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)


def remove_small_objects(
    mask: np.ndarray,
    min_size: int = 500,
) -> np.ndarray:
    """Remove connected components smaller than min_size voxels.

    Args:
        mask: Boolean 3-D array.
        min_size: Minimum number of voxels a component must have to survive.

    Returns:
        Cleaned boolean array with small components removed.
    """
    labeled, n_components = ndi.label(mask)
    logger.debug("Found %d connected components before size filtering", n_components)

    sizes = ndi.sum(mask, labeled, range(1, n_components + 1))
    keep = np.zeros_like(mask, dtype=bool)
    for idx, size in enumerate(sizes, start=1):
        if size >= min_size:
            keep |= labeled == idx

    n_removed = n_components - int(keep.any())
    logger.debug(
        "Removed %d component(s) smaller than %d voxels", n_removed, min_size
    )
    return keep


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in each 2-D axial slice of the mask.

    Operating slice-by-slice is much faster than 3-D hole-filling for large
    volumes and sufficient for closing gaps introduced by partial-volume
    effects at vessel boundaries.

    Args:
        mask: Boolean 3-D array (Z, Y, X convention from SimpleITK/NIfTI).

    Returns:
        Mask with holes filled per axial slice.
    """
    filled = np.empty_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        filled[z] = ndi.binary_fill_holes(mask[z])
    logger.debug("Hole-filling complete (%d slices processed)", mask.shape[0])
    return filled


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply binary dilation to ensure surface-voxel continuity.

    Args:
        mask: Boolean 3-D array.
        iterations: Number of dilation steps (default 1).

    Returns:
        Dilated boolean array.
    """
    dilated = ndi.binary_dilation(mask, iterations=iterations)
    logger.debug(
        "Dilation complete (%d iteration(s)); voxels: %d → %d",
        iterations,
        int(mask.sum()),
        int(dilated.sum()),
    )
    return dilated


def clean_mask(
    raw_mask_path: Path,
    output_path: Path,
    min_component_size: int = 500,
    fill: bool = True,
    dilation_iterations: int = 1,
) -> Path:
    """Full post-processing pipeline for a binary segmentation mask.

    Reads a NIfTI mask, applies remove_small_objects → fill_holes → dilate_mask,
    and writes the result back to output_path preserving image metadata.

    Args:
        raw_mask_path: Path to the raw combined_mask.nii.gz from segmentation.
        output_path: Destination path for the cleaned mask.
        min_component_size: Minimum voxel count for a component to survive.
        fill: Whether to run 2-D slice hole-filling (default True).
        dilation_iterations: Number of binary dilation iterations (default 1).

    Returns:
        Path to the written clean mask.

    Raises:
        ImportError: If SimpleITK is not installed.
        FileNotFoundError: If raw_mask_path does not exist.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is not installed. Install it with: pip install SimpleITK"
        ) from exc

    if not raw_mask_path.exists():
        raise FileNotFoundError(f"Raw mask not found: {raw_mask_path}")

    logger.info("Post-processing mask: %s", raw_mask_path)

    img = sitk.ReadImage(str(raw_mask_path))
    arr = sitk.GetArrayFromImage(img).astype(bool)

    logger.info("Input mask: %d foreground voxels", int(arr.sum()))

    # Step 1: remove small objects
    arr = remove_small_objects(arr, min_size=min_component_size)
    logger.info("After small-object removal: %d voxels", int(arr.sum()))

    # Step 2: fill holes slice-by-slice
    if fill:
        arr = fill_holes(arr)
        logger.info("After hole-filling: %d voxels", int(arr.sum()))

    # Step 3: dilate for surface continuity
    if dilation_iterations > 0:
        arr = dilate_mask(arr, iterations=dilation_iterations)
        logger.info("After dilation: %d voxels", int(arr.sum()))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(arr.astype(np.uint8))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(output_path))

    logger.info("Clean mask saved → %s", output_path)
    return output_path
