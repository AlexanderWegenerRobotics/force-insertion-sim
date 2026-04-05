"""
Decompose concave STLs into convex parts for MuJoCo collision.
Be carefull to setup the threshold and reolution properly. This needs to be a tradeoff between complexity and accuracy. Otherwise system might not enable insertion

Single file:
    python generate_hole_fixture.py meshes/raw/hole_cyl_l_loose.stl --output-root holes

Batch (all STLs in a folder):
    python generate_hole_fixture.py meshes/raw/ --output-root holes --batch
    python generate_hole_fixture.py meshes/raw/ --output-root holes --batch --threshold 0.01 --resolution 200
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
import coacd


def decompose_mesh(
    stl_path: str,
    output_root: str = "holes",
    threshold: float = 0.05,
    max_convex_hulls: int = -1,
    scale: float = 0.001,
    preprocess_resolution: int = 50,
) -> Path:
    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    stem = stl_path.stem

    fixture_dir = Path(output_root) / stem
    meshes_dir = fixture_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    visual_stl = meshes_dir / stl_path.name
    shutil.copy2(stl_path, visual_stl)
    print(f"  Copied visual mesh -> {visual_stl}")

    mesh = trimesh.load(stl_path)
    print(f"  Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    coacd.set_log_level("warn")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hulls,
        preprocess_resolution=preprocess_resolution,
    )
    print(f"  Decomposed into {len(parts)} convex parts")

    part_filenames = []
    for i, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        part_name = f"{stem}_col_{i}.stl"
        part_path = meshes_dir / part_name
        part_mesh.export(part_path)
        part_filenames.append(part_name)

    scale_str = f"{scale} {scale} {scale}"
    xml = _generate_xml(stem, part_filenames, scale_str)

    xml_path = fixture_dir / "fixture.xml"
    xml_path.write_text(xml)
    print(f"  Generated XML -> {xml_path}")

    return fixture_dir


def decompose_batch(
    input_dir: str,
    output_root: str = "holes",
    threshold: float = 0.05,
    max_convex_hulls: int = -1,
    scale: float = 0.001,
    preprocess_resolution: int = 50,
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    stl_files = sorted(input_dir.glob("*.stl")) + sorted(input_dir.glob("*.STL"))
    if not stl_files:
        print(f"No STL files found in {input_dir}")
        return []

    print(f"Found {len(stl_files)} STL files in {input_dir}\n")

    results = []
    for i, stl_path in enumerate(stl_files):
        print(f"[{i+1}/{len(stl_files)}] {stl_path.name}")
        try:
            fixture_dir = decompose_mesh(
                stl_path=str(stl_path),
                output_root=output_root,
                threshold=threshold,
                max_convex_hulls=max_convex_hulls,
                scale=scale,
                preprocess_resolution=preprocess_resolution,
            )
            results.append(fixture_dir)
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    print(f"Done! {len(results)}/{len(stl_files)} fixtures generated in {output_root}/")
    return results


def _generate_xml(
    stem: str,
    part_filenames: list[str],
    scale_str: str,
) -> str:
    asset_lines = []
    asset_lines.append(
        f'    <mesh name="{stem}_visual" file="meshes/{stem}.stl" scale="{scale_str}"/>'
    )
    for i, fname in enumerate(part_filenames):
        asset_lines.append(
            f'    <mesh name="{stem}_col_{i}" file="meshes/{fname}" scale="{scale_str}"/>'
        )
    assets = "\n".join(asset_lines)

    geom_lines = []
    geom_lines.append(
        f'      <geom name="{stem}_visual" type="mesh" mesh="{stem}_visual"\n'
        f'            contype="0" conaffinity="0"\n'
        f'            rgba="0.4 0.4 0.8 1.0"/>'
    )
    for i, _ in enumerate(part_filenames):
        geom_lines.append(
            f'      <geom name="{stem}_col_{i}" type="mesh" mesh="{stem}_col_{i}"\n'
            f'            rgba="0.4 0.4 0.8 0.0"\n'
            f'            friction="0.3 0.005 0.0001"\n'
            f'            condim="4"\n'
            f'            solref="0.005 1"\n'
            f'            solimp="0.95 0.999 0.002"\n/>'
        )
    geoms = "\n".join(geom_lines)

    xml = f"""\
<mujoco model="{stem}">
  <asset>
{assets}
  </asset>

  <worldbody>
    <body name="fixture" pos="0 0 0">
{geoms}

      <site name="hole_entrance" pos="0 0 0.1" size="0.002" rgba="0 1 0 1"/>
      <site name="hole_bottom"   pos="0 0 0.0" size="0.002" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def main():
    parser = argparse.ArgumentParser(
        description="Decompose concave STL(s) for MuJoCo collision."
    )
    parser.add_argument(
        "path", type=str,
        help="Path to a single STL file, or a directory of STLs (with --batch)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all STL files in the given directory",
    )
    parser.add_argument("--output-root", type=str, default="holes")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-hulls", type=int, default=-1)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--resolution", type=int, default=50)
    args = parser.parse_args()

    if args.batch:
        decompose_batch(
            input_dir=args.path,
            output_root=args.output_root,
            threshold=args.threshold,
            max_convex_hulls=args.max_hulls,
            scale=args.scale,
            preprocess_resolution=args.resolution,
        )
    else:
        fixture_dir = decompose_mesh(
            stl_path=args.path,
            output_root=args.output_root,
            threshold=args.threshold,
            max_convex_hulls=args.max_hulls,
            scale=args.scale,
            preprocess_resolution=args.resolution,
        )
        print(f"\nDone! Include with: <include file=\"{fixture_dir}/fixture.xml\"/>")


if __name__ == "__main__":
    main()