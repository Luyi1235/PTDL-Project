"""Geometry-only cross-surface proximity descriptors for VSPAERO meshes."""

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PAIRS = list(combinations(range(1, 7), 2))
KEYS = ("distance_min", "distance_q05", "distance_median", "relative_distance_min",
        "normal_alignment_near", "normal_separation_min", "shared_nodes_fraction")
FEATURE_NAMES = ["mesh_available"] + [f"pair_{a}_{b}_{key}"
                                     for a, b in PAIRS for key in KEYS]


def panel_proximity(path, area, aspect_ratio):
    lines = Path(path).read_text(encoding="utf8").splitlines()
    if not lines[0].startswith("# vspgeom") or int(lines[1]) != 1:
        raise ValueError("Expected one-block vspgeom mesh")
    n, count, _ = map(int, lines[2].split())
    xyz = np.array([[float(v) for v in row.split()] for row in lines[3:3+n]])
    offset = 3+n
    if int(lines[offset]) != count:
        raise ValueError("Incorrect mesh connectivity count")
    connectivity = [list(map(int, row.split())) for row in lines[offset+1:offset+1+count]]
    tags = [int(row.split()[0]) for row in lines[offset+1+count:offset+1+2*count]]
    groups = {tag: [] for tag in range(1, 7)}
    nodes = {tag: set() for tag in groups}
    for cell, tag in zip(connectivity, tags):
        if tag not in groups or cell[0] != len(cell)-1 or cell[0] < 3:
            raise ValueError("Unsupported surface or malformed polygon")
        p = xyz[np.array(cell[1:])-1]
        vector = np.cross(p, np.roll(p, -1, axis=0)).sum(0)/2
        magnitude = np.linalg.norm(vector)
        groups[tag].append((p.mean(0), vector/max(magnitude, 1e-15), np.sqrt(magnitude)))
        nodes[tag].update(cell[1:])
    chord = np.sqrt(area/aspect_ratio)
    if not np.isfinite(chord) or chord <= 0:
        raise ValueError("Positive geometry dimensions are required")
    values = [1.]
    for a, b in PAIRS:
        if not groups[a] or not groups[b]:
            values.extend([0.]*len(KEYS))
            continue
        centers_a, normals_a, lengths_a = map(np.asarray, zip(*groups[a]))
        centers_b, normals_b, lengths_b = map(np.asarray, zip(*groups[b]))
        distance, index = cKDTree(centers_b).query(centers_a)
        reverse, reverse_index = cKDTree(centers_a).query(centers_b)
        all_distance = np.r_[distance, reverse]/chord
        delta = centers_a-centers_b[index]
        normal_separation = abs(np.sum(delta*normals_b[index], axis=1))/chord
        near = distance <= np.quantile(distance, .1)
        alignment = abs(np.sum(normals_a*normals_b[index], axis=1))
        relative = np.r_[
            distance/np.maximum(lengths_a+lengths_b[index], 1e-12),
            reverse/np.maximum(lengths_b+lengths_a[reverse_index], 1e-12)]
        values.extend((all_distance.min(), np.quantile(all_distance, .05),
                       np.median(all_distance), relative.min(), alignment[near].mean(),
                       normal_separation.min(),
                       len(nodes[a] & nodes[b])/max(1, len(nodes[a] | nodes[b]))))
    result = np.asarray(values, float)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid panel proximity descriptors")
    return result
