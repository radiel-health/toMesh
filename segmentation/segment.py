"""
TotalSegmentator wrapper for cardiovascular structure segmentation.

Accepts DICOM folders or NIfTI files, converts DICOM to NIfTI via SimpleITK,
runs TotalSegmentator, merges target-structure masks, and saves outputs.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _require_totalsegmentator() -> None:
    """Fail fast with a helpful message if TotalSegmentator is not installed.

    Raises:
        ImportError: When totalsegmentator package is not found.
    """
    try:
        import totalsegmentator  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is not installed. Install it with:\n"
            "    pip install totalsegmentator\n"
            "and make sure the model weights are downloaded on first run."
        ) from exc


def _require_simpleitk() -> None:
    """Fail fast with a helpful message if SimpleITK is not installed.

    Raises:
        ImportError: When SimpleITK package is not found.
    """
    try:
        import SimpleITK  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is not installed. Install it with:\n"
            "    pip install SimpleITK"
        ) from exc


def dicom_to_nifti(dicom_dir: Path, output_path: Path) -> Path:
    """Convert a DICOM series folder to a single NIfTI file.

    Uses SimpleITK's ImageSeriesReader to read the series and writes a
    .nii.gz file. The first series found in the directory is used.
    TODO: add series-selection logic for scanners that write multiple series
    into the same directory.

    Args:
        dicom_dir: Directory containing DICOM (.dcm) files.
        output_path: Destination path for the output .nii.gz file.

    Returns:
        Path to the written NIfTI file.

    Raises:
        ImportError: If SimpleITK is not installed.
        ValueError: If no DICOM series is found in the directory.
    """
    _require_simpleitk()
    import SimpleITK as sitk

    logger.info("Converting DICOM folder %s → %s", dicom_dir, output_path)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    # Take first series; TODO: allow the caller to specify series ID
    series_id = series_ids[0]
    logger.debug("Using DICOM series ID: %s", series_id)

    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    logger.info("NIfTI written to %s", output_path)
    return output_path


def run_segmentation(
    input_path: Path,
    output_dir: Path,
    target_structures: List[str],
    task: str = "total",
    fast: bool = False,
) -> tuple[Path, dict[str, Path]]:
    """Run TotalSegmentator on a CT scan and return the combined mask.

    Workflow:
    1. Convert DICOM → NIfTI if input is a directory.
    2. Run TotalSegmentator with the specified task.
    3. For each target structure, load its individual mask.
    4. Merge all structure masks with logical OR into a combined mask.
    5. Save combined mask and individual masks to output_dir.

    Args:
        input_path: Path to a NIfTI file (.nii / .nii.gz) or a DICOM directory.
        output_dir: Directory where masks will be written.
        target_structures: List of TotalSegmentator structure names to include.
        task: TotalSegmentator task (default "total").
        fast: Whether to use TotalSegmentator's fast mode (lower accuracy).

    Returns:
        Tuple of:
            - combined_mask_path: Path to the saved combined_mask.nii.gz
            - individual_paths: Dict mapping structure name → individual mask path

    Raises:
        ImportError: If TotalSegmentator or SimpleITK are not installed.
        FileNotFoundError: If the input path does not exist.
        ValueError: If no DICOM series is found (DICOM input only).
    """
    _require_totalsegmentator()
    _require_simpleitk()

    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: resolve to NIfTI ------------------------------------------------
    if input_path.is_dir():
        nifti_path = output_dir / "ct_converted.nii.gz"
        nifti_path = dicom_to_nifti(input_path, nifti_path)
    else:
        nifti_path = input_path
        logger.info("Input is already NIfTI: %s", nifti_path)

    # --- Step 2: run TotalSegmentator --------------------------------------------
    seg_output_dir = output_dir / "totalsegmentator_output"
    seg_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running TotalSegmentator (task=%s, fast=%s) on %s", task, fast, nifti_path
    )
    totalsegmentator(
        input=str(nifti_path),
        output=str(seg_output_dir),
        task=task,
        fast=fast,
        # ml_backend="onnx"  # TODO: consider onnx backend for CPU-only environments
    )
    logger.info("TotalSegmentator finished. Output in %s", seg_output_dir)

    # --- Step 3: load individual structure masks ---------------------------------
    # TotalSegmentator writes one .nii.gz per structure, named exactly after the
    # structure label (e.g. "aorta.nii.gz").
    individual_paths: dict[str, Path] = {}
    combined_array: np.ndarray | None = None
    reference_img: "sitk.Image | None" = None

    for structure in target_structures:
        mask_file = seg_output_dir / f"{structure}.nii.gz"
        if not mask_file.exists():
            logger.warning(
                "Structure mask not found: %s (skipping). "
                "Check that '%s' is a valid TotalSegmentator label for task '%s'.",
                mask_file,
                structure,
                task,
            )
            continue

        logger.info("Loading mask for structure: %s", structure)
        img = sitk.ReadImage(str(mask_file))
        arr = sitk.GetArrayFromImage(img).astype(bool)

        # Save individual mask copy
        individual_out = output_dir / f"{structure}_mask.nii.gz"
        shutil.copy2(mask_file, individual_out)
        individual_paths[structure] = individual_out

        # Accumulate combined mask with logical OR
        if combined_array is None:
            combined_array = arr
            reference_img = img
        else:
            combined_array = combined_array | arr

    if combined_array is None or reference_img is None:
        raise RuntimeError(
            f"None of the requested structures were found in TotalSegmentator output. "
            f"Requested: {target_structures}. Check task and structure names."
        )

    # --- Step 4: save combined mask ----------------------------------------------
    combined_sitk = sitk.GetImageFromArray(combined_array.astype(np.uint8))
    combined_sitk.CopyInformation(reference_img)

    combined_mask_path = output_dir / "combined_mask_raw.nii.gz"
    sitk.WriteImage(combined_sitk, str(combined_mask_path))
    logger.info(
        "Combined mask saved (%d voxels set) → %s",
        int(combined_array.sum()),
        combined_mask_path,
    )

    return combined_mask_path, individual_paths
