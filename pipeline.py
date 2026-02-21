"""
toMesh end-to-end pipeline CLI.

Usage:
    python pipeline.py --input /path/to/ct.nii.gz --output /path/to/out/ --launch-gui
    python pipeline.py --input /path/to/ct/ --output /path/to/out/ --skip-gui
    python pipeline.py --input /path/to/ct.nii.gz --output /path/to/out/ \\
        --skip-segment --mask-path /path/to/combined_mask.nii.gz --launch-gui
    python pipeline.py --input /path/to/ct.nii.gz --output /path/to/out/ \\
        --export-only --mask-path /path/to/mesh_with_bcs.vtp

Flags:
    --input           Path to CT scan (NIfTI or DICOM folder)
    --output          Output directory
    --structures      Comma-separated TotalSegmentator structures (default: aorta,heart)
    --skip-segment    Skip segmentation; load existing mask from --mask-path
    --mask-path       Path to existing combined_mask.nii.gz (or .vtp for --export-only)
    --skip-gui        Headless mode — auto-detect BCs and export without GUI
    --launch-gui      Open the GUI after meshing
    --target-faces    Target face count for decimation (default: 15000)
    --export-only     Load existing .vtp with BC labels and re-export graph
    --config          Path to a custom config.yaml (default: ./config.yaml)
    --log-level       Logging verbosity: DEBUG, INFO, WARNING (default: INFO)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: Path) -> dict:
    """Load and return the YAML configuration.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Configuration dict. Returns an empty dict if the file is missing.
    """
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        logger.debug("Loaded config from %s", config_path)
        return cfg
    logger.warning("Config not found at %s; using defaults.", config_path)
    return {}


def run_segmentation_stage(
    input_path: Path,
    output_dir: Path,
    structures: List[str],
    cfg: dict,
) -> Path:
    """Run segmentation and mask cleaning; return the clean combined mask path.

    Args:
        input_path: CT scan path.
        output_dir: Where to write outputs.
        structures: List of TotalSegmentator structure names.
        cfg: Configuration dict from config.yaml.

    Returns:
        Path to the cleaned combined mask.
    """
    from segmentation.segment import run_segmentation
    from segmentation.postprocess import clean_mask

    seg_cfg = cfg.get("segmentation", {})
    task = seg_cfg.get("task", "total")

    logger.info("=== STAGE 1: SEGMENTATION ===")
    raw_mask_path, individual_paths = run_segmentation(
        input_path=input_path,
        output_dir=output_dir,
        target_structures=structures,
        task=task,
    )
    logger.info("Individual masks: %s", list(individual_paths.keys()))

    clean_cfg = cfg.get("segmentation", {})
    clean_path = output_dir / cfg.get("pipeline", {}).get("mask_filename", "combined_mask.nii.gz")

    clean_mask(
        raw_mask_path=raw_mask_path,
        output_path=clean_path,
        min_component_size=clean_cfg.get("min_component_size", 500),
        fill=clean_cfg.get("fill_holes", True),
        dilation_iterations=clean_cfg.get("dilation_iterations", 1),
    )
    return clean_path


def run_meshing_stage(
    mask_path: Path,
    output_dir: Path,
    cfg: dict,
    target_faces_override: Optional[int] = None,
) -> object:
    """Generate mesh from mask; return PyVista PolyData.

    Args:
        mask_path: Path to cleaned combined_mask.nii.gz.
        output_dir: Where to write mesh files.
        cfg: Configuration dict.
        target_faces_override: CLI override for target face count.

    Returns:
        PyVista PolyData of the generated mesh.
    """
    from meshing.generate_mesh import generate_mesh

    mesh_cfg = cfg.get("meshing", {})
    target_faces = target_faces_override or mesh_cfg.get("target_faces", 15_000)

    logger.info("=== STAGE 2: MESH GENERATION ===")
    mesh = generate_mesh(
        mask_path=mask_path,
        output_dir=output_dir,
        mc_level=mesh_cfg.get("mc_level", 0.5),
        target_faces=target_faces,
        max_hole_size=mesh_cfg.get("max_hole_size", 30),
        smoothing_iterations=mesh_cfg.get("smoothing_iterations", 5),
        smoothing_lambda=mesh_cfg.get("smoothing_lambda", 0.5),
        pyvista_smooth_iterations=mesh_cfg.get("pyvista_smooth_iterations", 20),
        pyvista_smooth_factor=mesh_cfg.get("pyvista_smooth_factor", 0.1),
        min_faces=mesh_cfg.get("min_faces", 1000),
        max_faces=mesh_cfg.get("max_faces", 200_000),
    )
    logger.info("Mesh generated: %d vertices, %d faces", mesh.n_points, mesh.n_faces_strict)
    return mesh


def run_headless_export(
    mesh_path: Path,
    output_dir: Path,
    cfg: dict,
    source_ct_path: Optional[Path] = None,
) -> None:
    """Auto-detect BCs and export graph without GUI.

    Args:
        mesh_path: Path to .vtp mesh file.
        output_dir: Export destination.
        cfg: Configuration dict.
        source_ct_path: Original CT scan path (for metadata).
    """
    import pyvista as pv
    import torch
    from gui.bc_tagger import ensure_bc_array, auto_detect_bc
    from export.to_graph import mesh_to_pyg
    from export.validators import validate_mesh_for_export

    logger.info("=== HEADLESS BC AUTO-DETECT + EXPORT ===")
    mesh = pv.read(str(mesh_path))
    mesh = ensure_bc_array(mesh)
    auto_detect_bc(mesh)

    export_cfg = cfg.get("export", {})
    min_nodes = export_cfg.get("min_nodes", 1000)
    max_nodes = export_cfg.get("max_nodes", 100_000)

    errors, warnings = validate_mesh_for_export(mesh, min_nodes=min_nodes, max_nodes=max_nodes)
    for w in warnings:
        logger.warning("Validation warning: %s", w)
    if errors:
        for e in errors:
            logger.error("Validation error: %s", e)
        logger.error("Export aborted due to validation errors.")
        return

    pipe_cfg = cfg.get("pipeline", {})
    graph_path = output_dir / pipe_cfg.get("graph_filename", "graph.pt")
    vtp_out = output_dir / "mesh_with_bcs.vtp"

    source_str = str(source_ct_path) if source_ct_path else ""
    radius_mult = export_cfg.get("neighbor_radius_multiplier", 3.0)
    data = mesh_to_pyg(mesh, source_file=source_str, neighbor_radius_multiplier=radius_mult)
    torch.save(data, str(graph_path))
    mesh.save(str(vtp_out))

    logger.info("Headless export complete:")
    logger.info("  Graph: %s", graph_path)
    logger.info("  Mesh:  %s", vtp_out)


def run_export_only(
    vtp_path: Path,
    output_dir: Path,
    cfg: dict,
    source_ct_path: Optional[Path] = None,
) -> None:
    """Load an existing labelled .vtp and re-export the graph.

    Args:
        vtp_path: Path to a .vtp file that already has bc_label.
        output_dir: Export destination.
        cfg: Configuration dict.
        source_ct_path: Original CT scan path.
    """
    import pyvista as pv
    import torch
    from export.to_graph import mesh_to_pyg
    from export.validators import validate_mesh_for_export

    logger.info("=== EXPORT ONLY: loading %s ===", vtp_path)
    mesh = pv.read(str(vtp_path))

    export_cfg = cfg.get("export", {})
    errors, warnings = validate_mesh_for_export(
        mesh,
        min_nodes=export_cfg.get("min_nodes", 1000),
        max_nodes=export_cfg.get("max_nodes", 100_000),
    )
    for w in warnings:
        logger.warning("%s", w)
    if errors:
        for e in errors:
            logger.error("%s", e)
        sys.exit(1)

    pipe_cfg = cfg.get("pipeline", {})
    graph_path = output_dir / pipe_cfg.get("graph_filename", "graph.pt")
    radius_mult = export_cfg.get("neighbor_radius_multiplier", 3.0)
    source_str = str(source_ct_path) if source_ct_path else ""

    data = mesh_to_pyg(mesh, source_file=source_str, neighbor_radius_multiplier=radius_mult)
    torch.save(data, str(graph_path))
    logger.info("Graph saved: %s", graph_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--input", "input_path", required=True, type=click.Path(exists=True),
    help="CT scan path (NIfTI or DICOM folder).",
)
@click.option(
    "--output", "output_dir", required=True, type=click.Path(),
    help="Output directory.",
)
@click.option(
    "--structures", default="aorta,heart",
    help="Comma-separated TotalSegmentator structure names.",
)
@click.option(
    "--skip-segment", is_flag=True, default=False,
    help="Skip segmentation; use --mask-path.",
)
@click.option(
    "--mask-path", "mask_path", default=None, type=click.Path(),
    help="Existing mask (.nii.gz) or labelled mesh (.vtp) path.",
)
@click.option(
    "--skip-gui", is_flag=True, default=False,
    help="Headless: auto-detect BCs and export without GUI.",
)
@click.option(
    "--launch-gui", is_flag=True, default=False,
    help="Open the mesh editor GUI after generating the mesh.",
)
@click.option(
    "--target-faces", default=15_000, show_default=True,
    help="Target face count for mesh decimation.",
)
@click.option(
    "--export-only", is_flag=True, default=False,
    help="Load existing labelled .vtp and re-export graph.",
)
@click.option(
    "--config", "config_path",
    default=str(Path(__file__).parent / "config.yaml"),
    type=click.Path(),
    help="Path to config.yaml.",
)
@click.option(
    "--log-level", default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def main(
    input_path: str,
    output_dir: str,
    structures: str,
    skip_segment: bool,
    mask_path: Optional[str],
    skip_gui: bool,
    launch_gui: bool,
    target_faces: int,
    export_only: bool,
    config_path: str,
    log_level: str,
) -> None:
    """toMesh — end-to-end cardiovascular mesh pipeline."""
    _setup_logging(log_level)

    input_p = Path(input_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(Path(config_path))
    structure_list = [s.strip() for s in structures.split(",") if s.strip()]

    logger.info("toMesh pipeline starting")
    logger.info("  Input:      %s", input_p)
    logger.info("  Output:     %s", output_p)
    logger.info("  Structures: %s", structure_list)

    # -- EXPORT ONLY mode -------------------------------------------------
    if export_only:
        if not mask_path:
            logger.error("--export-only requires --mask-path pointing to a .vtp file.")
            sys.exit(1)
        run_export_only(
            vtp_path=Path(mask_path),
            output_dir=output_p,
            cfg=cfg,
            source_ct_path=input_p,
        )
        return

    # -- Resolve mask path ------------------------------------------------
    pipe_cfg = cfg.get("pipeline", {})
    if skip_segment:
        if not mask_path:
            logger.error("--skip-segment requires --mask-path.")
            sys.exit(1)
        clean_mask_path = Path(mask_path)
        logger.info("Skipping segmentation; using mask: %s", clean_mask_path)
    else:
        clean_mask_path = run_segmentation_stage(
            input_path=input_p,
            output_dir=output_p,
            structures=structure_list,
            cfg=cfg,
        )

    # -- Mesh generation --------------------------------------------------
    mesh = run_meshing_stage(
        mask_path=clean_mask_path,
        output_dir=output_p,
        cfg=cfg,
        target_faces_override=target_faces,
    )

    vtp_path = output_p / pipe_cfg.get("mesh_vtp_filename", "mesh.vtp")

    # -- Headless export or GUI launch ------------------------------------
    if skip_gui:
        run_headless_export(
            mesh_path=vtp_path,
            output_dir=output_p,
            cfg=cfg,
            source_ct_path=input_p,
        )
    elif launch_gui:
        logger.info("Launching GUI with mesh: %s", vtp_path)
        from gui.main_window import launch_gui as _launch_gui
        exit_code = _launch_gui(
            mesh_path=vtp_path,
            output_dir=output_p,
            source_ct_path=input_p,
        )
        # After GUI exits, check if a graph was exported
        graph_path = output_p / pipe_cfg.get("graph_filename", "graph.pt")
        if graph_path.exists():
            logger.info("Graph exported: %s", graph_path)
        else:
            logger.info("GUI closed without exporting a graph.")
        sys.exit(exit_code)
    else:
        logger.info(
            "Pipeline complete. Mesh at %s. "
            "Use --launch-gui or --skip-gui to proceed.", vtp_path
        )


if __name__ == "__main__":
    main()
