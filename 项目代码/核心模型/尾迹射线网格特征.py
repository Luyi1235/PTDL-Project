"""Geometry-only wake-ray descriptors for the VSPAERO v3 mesh format."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANGLES = (-3., -1., 1., 3., 5.)
PATCHES = tuple(range(1, 7))
SHEETS = ((2, 3), (4, 5), (6,))
STATISTICS = ("near_distance", "distance_q05", "influence_max", "influence_rms",
              "influence_signed_mean", "influence_abs_mean", "near_fraction")
FEATURE_NAMES = (
    ["mesh_available", "node_count", "panel_count", "wake_sheet_count"]
    + [f"patch_{tag}_panels" for tag in PATCHES]
    + [f"sheet_{i}_nodes" for i in range(len(SHEETS))]
    + [f"angle_{a:g}_sheet_{i}_patch_{tag}_{key}"
       for a in ANGLES for i in range(len(SHEETS)) for tag in PATCHES for key in STATISTICS]
)


@dataclass(frozen=True)
class WakeMesh:
    xyz: np.ndarray
    cells: tuple
    tags: np.ndarray
    sheets: dict


def read_wake_mesh(path):
    lines = Path(path).read_text(encoding="utf8").splitlines()
    if lines[0].strip() != "# vspgeom v3" or int(lines[1]) != 1:
        raise ValueError("Expected one-block VSPAERO v3 geometry")
    n, count, sheets_count = map(int, lines[2].split())
    xyz = np.array([list(map(float, row.split())) for row in lines[3:3+n]])
    offset = 3+n
    if xyz.shape != (n, 3) or int(lines[offset]) != count:
        raise ValueError("Malformed mesh coordinate or panel count")
    cells = []
    for line in lines[offset+1:offset+1+count]:
        row = list(map(int, line.split()))
        if row[0] != len(row)-1 or row[0] < 3 or min(row[1:]) < 1 or max(row[1:]) > n:
            raise ValueError("Malformed mesh connectivity")
        cells.append(np.asarray(row[1:], int)-1)
    tags = np.array([int(row.split()[0])
                     for row in lines[offset+1+count:offset+1+2*count]])
    cursor = offset+1+3*count
    if len(tags) != count or int(lines[cursor]) != sheets_count:
        raise ValueError("Malformed mesh wake-sheet section")
    # Wake node lists can wrap across lines. Later sections are left unread.
    tokens = iter(" ".join(lines[cursor+1:]).split())
    sheets = {}
    for _ in range(sheets_count):
        size, tag = int(next(tokens)), int(next(tokens))
        indices = np.array([int(next(tokens))-1 for _ in range(size)], int)
        if size < 2 or min(indices) < 0 or max(indices) >= n or tag in sheets:
            raise ValueError("Malformed trailing-edge nodes")
        sheets[tag] = indices
    if set(tags)-set(PATCHES) or set(sheets)-{2, 3, 4, 5, 6} or not np.isfinite(xyz).all():
        raise ValueError("Unexpected airframe mesh surfaces")
    return WakeMesh(xyz, tuple(cells), tags, sheets)


def ray_descriptors(mesh, wing_area, aspect_ratio):
    chord = np.sqrt(wing_area/aspect_ratio)
    if not np.isfinite(chord) or chord <= 0:
        raise ValueError("Positive reference dimensions are required")
    xyz = (mesh.xyz-mesh.xyz.mean(0))/chord
    centers, normals = [], []
    for cell in mesh.cells:
        vertices = xyz[cell]
        vector = np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(0)/2
        centers.append(vertices.mean(0))
        normals.append(vector/max(np.linalg.norm(vector), 1e-15))
    centers, normals = np.asarray(centers), np.asarray(normals)
    sources = []
    for group in SHEETS:
        indices = np.unique(np.concatenate([mesh.sheets[tag] for tag in group
                                            if tag in mesh.sheets]))
        sources.append(xyz[indices])
    values = [1., len(xyz), len(mesh.cells), len(mesh.sheets)]
    values.extend(float(np.sum(mesh.tags == tag)) for tag in PATCHES)
    values.extend(float(len(s)) for s in sources)
    for angle in ANGLES:
        alpha = np.deg2rad(angle)
        direction = np.array([np.cos(alpha), 0., np.sin(alpha)])
        for source in sources:
            delta = centers[:, None, :]-source[None, :, :]
            along = delta @ direction
            radial2 = np.maximum(np.sum(delta*delta, axis=2), 1e-20)
            cross = np.cross(direction, delta)
            perpendicular2 = np.maximum(np.sum(cross*cross, axis=2), 1e-16)
            ray_distance = np.sqrt(perpendicular2+np.minimum(along, 0.)**2)
            # This geometric kernel has unit strength and contains no solved circulation.
            kernel = cross/perpendicular2[:, :, None] * (
                1+along/np.sqrt(radial2))[:, :, None]/(4*np.pi)
            coupling = np.sum(kernel*normals[:, None, :], axis=2)
            for tag in PATCHES:
                mask = mesh.tags == tag
                distance, influence = ray_distance[mask], coupling[mask]
                if not len(distance):
                    values.extend([0.]*len(STATISTICS))
                    continue
                absolute = abs(influence)
                mean = float(influence.mean())
                values.extend((
                    np.log1p(1/max(float(distance.min()), 1e-12)),
                    np.log1p(1/max(float(np.quantile(distance, .05)), 1e-12)),
                    np.log1p(absolute.max()),
                    np.log1p(np.sqrt(np.mean(influence**2))),
                    np.sign(mean)*np.log1p(abs(mean)),
                    np.log1p(absolute.mean()),
                    np.mean(distance < .02),
                ))
    result = np.asarray(values, float)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid wake-ray descriptors")
    return result


def extract(path, wing_area, aspect_ratio):
    return ray_descriptors(read_wake_mesh(path), wing_area, aspect_ratio)
